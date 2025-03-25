import datetime
import pathlib
import subprocess
import os
import shutil
import tarfile

from typing import List, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from plot import plot_dir
from data import *


class TestSuiteItem(BaseModel):
        test_name: str = Field(description="Identifier for the specific test")
        test_type: str = Field(description="Type of measurement to perform")
        collective: str = Field(description="Collective program to run")
        messages_data: Union[str, dict] = Field(description="Filename of messages from data.py or function parameters")
        timeout: Optional[int] = Field(default=1, description="Timeout for individual tests")


class GlobalConfigOutput(BaseModel):
        directory: Optional[str] = Field(default=None, description="Directory to save results and parameters to")
        verbose: Optional[bool] = Field(default=None, description="Verbose output")


class GlobalConfig(BaseModel):
        max_runtime: Optional[int] = Field(default=None, ge=0, description="Maximum runtime in seconds for each test")
        nproc: Optional[int] = Field(default=None, ge=2, description="Number of processes")
        output: Optional[GlobalConfigOutput] = None


class OpenMPIBenchmarkConfig(BaseModel):
        benchmark_name: str = Field(description="Name of the benchmark configuration")
        test_suite: List[TestSuiteItem]
        global_config: Optional[GlobalConfig]

        class Config:
                str_min_length = 1  # Ensures strings are not empty
                str_strip_whitespace = True  # Trims whitespace from strings


def generate_data_file(data: str, params: dict):
        if data == 'equal':
                return equal(**params)
        elif data == 'normal':
                return normal(**params)
        elif data == 'exponential':
                return exponential(**params)
        elif data == 'increasing':
                return increasing(**params)
        elif data == 'decreasing':
                return decreasing(**params)
        elif data == 'zipfian':
                return zipfian(**params)
        elif data == 'uniform':
                return uniform(**params)
        elif data == 'bucket':
                return bucket(**params)
        elif data == 'spikes':
                return spikes(**params)
        elif data == 'alternating':
                return alternating(**params)
        elif data == "two_blocks":
                return two_blocks(**params)
        else:
                raise ValueError(f"Unknown test name {data}")


def parse_file(filename: str):
        try:
                json_string = pathlib.Path(filename).read_text()
                return OpenMPIBenchmarkConfig.model_validate_json(json_string, strict=False)
        except ValidationError as e:
                print("==> Validation error")
                print(e)
                raise SystemExit(1)
        except FileNotFoundError as e:
                print("==> Could not find program")
                print(e)
        raise SystemExit(1)


def create_output(dirname: str, no_save: bool, verbose: bool, savename: str):
        if no_save:
                return pathlib.Path().cwd()

        output = pathlib.Path(dirname)
        if not output.exists():
                output.mkdir(parents=True)

        if savename is not None:
                output = output / savename
        else:
                name = f"results-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
                output = output / name

        if output.exists():
                i = len(list(output.parent.glob(f"{str(output)}*"))) + 1
                output = pathlib.Path(dirname) / f"{str(output)}-{i}"

        output.mkdir(parents=True)
        if verbose:
                print(f"==> Created output directory: {output}")
        return output


def parse_message_data(messages_data: Union[str, dict], output: pathlib.Path):
        if isinstance(messages_data, str):
                # Exisiting file
                return messages_data
        else:
                # Generate file based on test name and params
                data = messages_data.get("data")
                params = messages_data.get("params")
                params["savedir"] = output

                if not data or not params:
                        print("==> Invalid function parameters.")
                        raise SystemExit(1)

                _, messages_data = generate_data_file(data, params)
                return messages_data


def schedule_script(collective: str, executor: str, nproc: str, mpi_impl: str):
        cmd = "#!/bin/bash\n"
        if "srun" in executor:
                impl = "openmpi@4.1.6" if mpi_impl == "openmpi" else "mpich@4.1.2"

                cmd += f"#SBATCH --job-name=mpi_job                      # Name of the job\n"
                cmd += f"#SBATCH --output=openmpi_output.txt             # Standard output file\n"
                cmd += f"#SBATCH --error=openmpi_error.txt               # Standard error file\n"
                cmd += f"#SBATCH --ntasks={nproc}                        # Number of tasks (processes)\n"
                cmd += f"#SBATCH --time=03:00:00                         # Max runtime (3 hour)\n"
                cmd += f"spack load {impl}\n"
                cmd += f"srun {collective}\n"

        elif "mpirun" in executor:
                cmd += f"mpirun.{mpi_impl} -np {nproc} {collective}\n"

        return cmd


def run_test(command):
        env = os.environ.copy()
        try:
                subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as e:
                print("==> Got non-zero error code")
                print(e)
                raise SystemExit(1)

def main(filename: str,
         executor: str,
         wd: str = ".",
         mpi_impl: str = 'openmpi',
         compress: bool = True,
         plot: bool = True):
        no_save = not (compress or plot)
        start = datetime.datetime.now()

        benchmark = parse_file(filename)
        verbose = benchmark.global_config.output.verbose
        nproc = benchmark.global_config.nproc

        cwd = pathlib.Path(wd)
        savename = f"{benchmark.benchmark_name}-{mpi_impl}"
        output = create_output(dirname=benchmark.global_config.output.directory,
                               savename=savename,
                               no_save=no_save,
                               verbose=verbose)

        for test in benchmark.test_suite:
                now = datetime.datetime.now()
                diff = (now - start).total_seconds()
                if diff > benchmark.global_config.max_runtime:
                        print(f"==> Max runtime reached: EXIT")
                        raise SystemExit(1)

                # Override globally set nproc if set in test case
                if isinstance(test.messages_data, dict) and "nproc" in test.messages_data.get("params").keys():
                        nproc = test.messages_data.get("params")["nproc"]
                elif nproc is None:
                        print("==> Number of processes required.")
                        raise SystemExit(1)

                end = "\n" if verbose else "\t\t\t\t"
                if verbose:
                        print(f"==> Started {test.test_name}", end=end, flush=True)

                # Get or create message data
                messages_data = parse_message_data(messages_data=test.messages_data, output=output)

                # Name of file where latencies will be saved to
                foutput = output / f"{test.test_name}.csv" if not no_save else f"{test.test_name}.csv"

                collective_call = f"{str(cwd.absolute() / test.collective)} "
                collective_call += f"--fmessages {messages_data} "
                collective_call += f"--foutput {foutput} "
                collective_call += f"--timeout {test.timeout} "
                if verbose:
                        collective_call += "--verbose "

                # Build script to run
                script = schedule_script(
                        executor=str(executor),
                        nproc=str(nproc),
                        mpi_impl=str(mpi_impl),
                        collective=str(collective_call)
                )
                script_save = output / f"{test.test_name}.slurm"
                script_save.write_text(script, encoding="utf8")
                script_save.chmod(script_save.stat().st_mode | os.X_OK)

                # Execute the script
                cmd = "bash " + str(script_save.absolute())
                run_test(cmd)

                # Remove temporary messages file
                if no_save:
                        pathlib.Path(messages_data).unlink()
                        pathlib.Path(foutput).unlink()

        if plot:
                if verbose:
                        print(f"==> Plotting test results ... ", end="", flush=True)
                plot_dir(dirname=str(output))
                if verbose:
                        print("Done")

        if compress:
                if verbose:
                        print(f"==> Compressing {str(output)}.tar.gz ... ", end="", flush=True)
                tar = tarfile.open(f"{str(output)}.tar.gz", "w:xz")
                tar.add(output)
                tar.close()
                if verbose:
                        print("Done")

        if verbose:
                now = datetime.datetime.now()
                diff = (now - start).total_seconds()
                print(f"==> Completed. Required {diff} seconds.")


if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("filename")
        parser.add_argument("--ask",
                            action='store_true',
                            default=False,
                            help="If set will ask for output directory name (default: False)")
        parser.add_argument("--no-compress",
                            action='store_false',
                            default=True,
                            help="If set will not create tar.xz for output directory (default=False)")
        parser.add_argument("--no-plot",
                            action='store_false',
                            default=True,
                            help="If set will not create plots of runs (default=False)")
        parser.add_argument("--executor",
                            default="mpirun",
                            choices=["mpirun", "srun"],
                            help="The job scheduler to use (default: mpirun)")
        parser.add_argument("--mpi-impl",
                            default="openmpi",
                            choices=["openmpi", "mpich"],
                            help="MPI implementation to use (default: openmpi)")
        parser.add_argument("--wd",
                            default='.',
                            help="Working directory with binaries (default: .)")
        args = parser.parse_args()

        e = shutil.which(args.executor)
        assert e is not None, f"{args.executor} was not found in path"

        main(filename=args.filename,
             executor=e,
             compress=args.no_compress,
             mpi_impl=args.mpi_impl,
             plot=args.no_plot,
             wd=args.wd)
