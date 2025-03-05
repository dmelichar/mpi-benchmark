# This is basically a fancy mpirun wrapper.
# Alternatively, one could have simple shell scripts that would
# be the _testcases_ or in other words the different benchmarks to run

import datetime
import pathlib
import argparse
import subprocess
import os
import shutil
import sys

from typing import List, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from data import equal, normal, exponential, increasing, decreasing, zipfian, uniform, bucket, spikes, alternating, \
        two_blocks


class TestSuiteItem(BaseModel):
        test_name: str = Field(description="Identifier for the specific test")
        test_type: str = Field(description="Type of measurement to perform")
        collective: str = Field(description="Collective program to run")
        messages_data: Union[str, dict] = Field(description="Filename of messages from data.py or function parameters")
        timeout: Optional[int] = Field(default=1, description="Timeout for individual tests")


class GlobalConfigOutput(BaseModel):
        directory: str = Field(default="./results", description="Directory to save results and parameters to")
        verbose: bool = Field(default=False, description="Verbose output")


class GlobalConfig(BaseModel):
        max_runtime: int = Field(ge=0, description="Maximum runtime in seconds for each test")
        nproc: int = Field(ge=2, description="Number of processes")
        trials: Optional[int] = Field(default=1, description="Number of times to run all tests")
        output: GlobalConfigOutput = None


class OpenMPIBenchmarkConfig(BaseModel):
        benchmark_name: str = Field(description="Name of the benchmark configuration")
        test_suite: List[TestSuiteItem]
        global_config: GlobalConfig

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


def main(filename: str, executor: str, ask: bool = False):
        try:
                json_string = pathlib.Path(filename).read_text()
                benchmark = OpenMPIBenchmarkConfig.model_validate_json(json_string, strict=True)
                env = os.environ.copy()

                start = datetime.datetime.now()
                verbose = benchmark.global_config.output.verbose
                output = pathlib.Path(benchmark.global_config.output.directory)
                if ask and output.exists():
                        print(f"==> Output directory >{output}< exists. Overwrite? [y/n]")
                        overwrite = input()
                        if overwrite == "n":
                                output = output.parent / f"results-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
                        elif overwrite != "y":
                                print("==> Unknown input. Abort")
                                sys.exit()
                else:
                        output = output.parent / f"results-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

                # Ensure the directory is created
                output.mkdir(parents=True)
                print(f"==> Created output directory: {output}")

                for test in benchmark.test_suite:
                        end = "\n" if verbose else "\t\t\t\t"
                        print(f"==> Started {test.test_name}", end=end, flush=True)
                        cwd = pathlib.Path().cwd()
                        op = cwd / test.collective
                        op = cwd / op

                        nproc = str(benchmark.global_config.nproc)

                        if isinstance(test.messages_data, str):
                                messages_data = test.messages_data
                        else:
                                # Generate file based on test name and params
                                data = test.messages_data.get("data")
                                params = test.messages_data.get("params")
                                params["savedir"] = str(output)

                                if not data or not params:
                                        print("==> Invalid function parameters.")
                                        continue

                                _, messages_data = generate_data_file(data, params)

                                if "nproc" in params.keys():
                                        nproc = str(params["nproc"])

                        for i in range(benchmark.global_config.trials):
                                now = datetime.datetime.now()
                                diff = (now - start).total_seconds()
                                if diff > benchmark.global_config.max_runtime:
                                        print(f"==> Max runtime reached: EXIT")
                                        sys.exit()

                                foutput = output / f"{test.test_name}-{i + 1}.csv"

                                executor_command = []
                                if executor == "srun":
                                        executor_command.append(executor)
                                        executor_command.append(f"-N{nproc}")
                                        executor_command.append("--ntasks-per-node=1")
                                elif executor == "mpirun":
                                        executor_command.append(executor)
                                        executor_command.append("-np")
                                        executor_command.append(nproc)

                                mpi_command = [str(op),
                                               "--fmessages", str(messages_data),
                                               "--foutput", str(foutput),
                                               "--timeout", str(test.timeout),
                                               "--verbose" if verbose else ""
                                               ]
                                command = executor_command + mpi_command
                                print(f"==> Trial [{i + 1} / {benchmark.global_config.trials}]", end=end, flush=True)
                                result = subprocess.run(command,
                                                        check=True,
                                                        env=env,
                                                        )

                                if result.returncode != 0:
                                        print("==> FAILED. Next test.")
                                        break

        except ValidationError as e:
                print("==> Validation error")
                print(e)
                raise SystemExit(1)
        except FileNotFoundError as e:
                print("==> Could not find program")
                print(e)
                raise SystemExit(1)
        except subprocess.CalledProcessError as e:
                print("==> Got non-zero error code")
                print(e)
                raise SystemExit(1)
        except Exception as e:
                print("==> Other exception")
                print(e)
                raise SystemExit(1)


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("filename")
        parser.add_argument("--ask", action='store_true', help="If false will create output directory")
        parser.add_argument("--executor", default="mpirun", help="The executor to run: mpirun or srun")
        args = parser.parse_args()

        assert shutil.which(args.executor) is not None, f"{args.executor} was not found in path"

        main(filename=args.filename, ask=args.ask, executor=args.executor)
