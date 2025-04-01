# MPI Benchmark

> An implementation of MPI collectives and a benchmark suite to test their latency performance

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Message distribution](#message-distribution)
- [Test suite](#test-suite)

## Overview

This repository contains source code that allows for testing different implementations of the MPI standard. The repository consists of a few programs that each do one thing well and can be used together. The repository is structured as follows

    ├── CMakeLists.txt
    ├── README.md
    ├── src/
       ├── allgatherv.cpp
       ├── alltoallw.cpp
       ├── bcast.cpp
       ├── gatherv.cpp
       └── scatterv.cpp
    └──  test/
        ├── scatterv/
        ├── gatherv/
        ├── allgatherv/
        ├── alltoallw/
        ├── custom.csv
        ├── data.py
        ├── example.json
        ├── main.ipynb
        ├── plot.py
        ├── requirements.txt
        └── suite.py

We are using CMake 3.25.1 as build system but require a minimum version of 3.18.4 to build the source files. The `CMakeLists.txt` contains the configuration and build instructions for the project, including the import of source files in `src/`, compiler options, and dependencies, which allows CMake to generate platform-specific build files. We use C++20 to implement the benchmark of a subset of the MPI [collective operations](https://en.wikipedia.org/wiki/Collective_operation) that can be found in `src/` and Python 3.9 to run a suite of tests to benchmark their latencies that can be found in `test/`.

The benchmark files in `src/` essentially call the respective MPI collective operation as often as possible within a specified time limit. For example, if the `scatterv` binary is called with a time limit set to one second then as many `scatterv` calls are placed as possible within that second. The benchmark also includes a feature to define the count of messages to be sent by each process to each other process through a CSV file, which we will now refer to as the message distribution. We provide code to generate the message distribution with some common patterns. The code reads the message distribution CSV file, and it runs for a specified duration. The latency results are gathered from all processes, calculated locally, and then reduced to show global minimum, maximum, and average latencies. Finally, the program saves the latencies to an output file for further analysis. 

The test files in `test/` consist of `data.py` which allows for generating  some common message distribution CSV files, `suite.py` which allows for running tests on the binaries with different message distributions, and `plot.py` which shows some basic graphics for the latency results of tests. 

The `test/custom.csv` file is an example of how a message distribution for either `scatterv`, `gatherv`, or `allgatherv` with four processes looks like. It reads as the root process sending 10 messages to each other process including itself. The message distribution will be explained in the [message distribution](#message-distribution) section.

The `test/example.json` file is an example of how a suite of test cases looks like. It consists of three test cases each with a different message distribution that will be generated when the test suite is run. It also specifies the number of processes to run, the timeout after which the test suite will stop, and where the latency results shall be saved to. We also provide a test cases for each of the common message distributions defined in `data.py`. The test suite will be explained in more detail in the [test suite](#test-suite) section.

## Installation

### Prerequisites

- CMake 3.18.4
- C++20
- Python 3.9
- OpenMPI 4.1.6 or MPICH 4.1.2

### Setup instructions

1. Clone the repository
  ``` bash
  git clone https://github.com/dmelichar/mpi-benchmark
  ```
2. (optional) It may be necessary to load the desired MPI implementation, for example
  ``` bash
  spack load openmpi@4.1.6  # or
  spack load mpich@4.1.2
  ```
3. Build the binaries
  ``` bash
  mkdir mpi-benchmark/build
  cd mpi-benchmark/build
  cmake ..
  make
  ```
4. (optional) Verify that the binary is built with the desired MPI implementation 
  ``` bash
  ldd /scatterv | grep mpi
  ```
5. Setup virtual test environment
  ``` bash
  cd mpi-benchmark/test
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

## Usage

In the following we will describe a usage workflow for `scatterv`. Once the binaries are build with CMake and the virtual environment is setup we either run tests on the binaries themselves or use `test/suite.py` to run multiple test cases in sequence. All binaries `scatterv`, `gatherv`, `allgatherv`, `alltoallw` accept the same input parameters:

``` bash
./scatterv --help   
Help: This program runs a MPI scatterv
Options:
  -h, --help            Show this help message
  -m, --fmessages FILE  Specify file with messages (default: default_messages.txt)
  -o, --foutput FILE    Specify output file (default: default_output.txt)
  -t, --timeout NUM     Specify timeout value in seconds (default: 10)
  -d, --dtype TYPE      Specify char for MPI_CHAR, double MPI_DOUBLE or int for MPI_INT32 (default: double)
  -v, --verbose         Enable verbose mode
```

All binaries require a message distribution file to be passed with the `-m` or `--fmessages` option, which can be generated with `data.py`. For example, to generate a distribution for four process where each process recives 10 messages from root we can use

``` bash
python data.py equal --seed 42 4 10
```

which will generate a file called `4-equal.csv`. 

An example run of `scatterv` would be the following

``` bash
mpirun -np 4 scatterv \
  --fmessages 4-equal.csv \
  --foutput scatterv-latencies.txt \
  --timeout 1 \
  --dtype char 
```

Note that we also specify the data type of the messages to be sent by root.

## Message distribution

The `data.py` file generates a CSV file that encodes how many messages are to be send and/or received by each process. It considers the case of one-to-many collective operations such as `Scatterv` where each process receives messages from one root process and the case of many-to-many collective operations such as `Alltoall` where each process sends messages and receives messages.

We provide the following distributions:

- **equal**: Generates a uniform distribution of a specified value (val) across a specified number of processors (nproc). Optionally supports a many-to-many distribution (m2m). It follows that for each $$m_i = val$$ with $i = 0, 1, ..., nrpoc-1$
- **normal**: Generates a normal distribution with a mean of 10 and standard deviation of 15, for a specified number of processors (nproc). Optionally supports a many-to-many distribution (m2m). It follows that for each $$m_i \sim N(10, 15)$$ with $i = 0, 1, ..., nrpoc-1$
- **exponential**: Generates an exponential distribution with a mean of 50 for the specified number of processors (nproc). Optionally supports a many-to-many distribution (m2m). It follows that for each $$m_i \sim exp(50)$$ with $i = 0, 1, ..., nrpoc-1$
- **increasing**: Generates an increasing sequence of block sizes based on a specified average block size (avg) and number of processors (nproc). It follows that for each message $$m_i = \left \lfloor \frac{2\mathop{avg}*(i+1)}{\mathop{nproc}-1} \right \rfloor$$ with $i = 0,1,...,nproc-1$ 
- **decreasing**: Generates a decreasing sequence of block sizes based on a specified average block size (avg) and number of processors (nproc). It follows that for each message $$m_i = \left \lfloor \frac{2\mathop{avg}*(\mathop{nproc}-1)}{\mathop{nproc}} \right \rfloor+1$$ with $i = 0, 1, ..., nproc-1$
- **zipfian**: Generates a Zipfian (or Zeta) distribution for the specified number of processors (nproc). Optionally supports a many-to-many distribution (m2m)-
- **uniform**: Generates a uniform distribution of random block sizes around a specified average (avg) for the specified number of processors (nproc). Optionally supports a many-to-many distribution (m2m). It follows that for each $$m_i \in  [1, 2 * \mathop{avg}] $$ with $i = 0, 1, ..., nrpoc-1$
- **bucket**: Generates block sizes based on a bucket distribution, where the block sizes range between half and the full specified average (avg), for the specified number of processors (nproc). Optionally supports a many-to-many distribution (m2m). It follows that for each $$m_i = \frac{\mathop{avg}}{2} + x$$ with $i = 0, 1, ..., nrpoc-1$ and $x \in [1, \mathop{avg}]$
- **spikes**: Generates block sizes with spike values based on a specified average (avg) and spike factor (rho). The block sizes alternate between avg multiplied by rho and 1, for the specified number of processors (nproc). Optionally supports a many-to-many distribution (m2m). It follows that for each $$m_i = \begin{cases}\rho * \mathop{avg} && \text{with probability } \frac{1}{\rho} \\ 1 && \text{with probability } 1-\frac{1}{\rho}\end{cases}$$  with $i = 0, 1, ..., nrpoc-1$
- **alternating**: Generates alternating block sizes between avg + avg // 2 and avg - avg // 2 for the specified number of processors (nproc). Optionally supports a many-to-many distribution (m2m). It follows that for each $$m_i = \begin{cases}\mathop{avg} + \frac{\mathop{avg}}{2} && \text{if } i \text{ is even} \\ \mathop{avg} - \frac{\mathop{avg}}{2} && \text{if } i \text{ is odd}\end{cases} $$ with $i = 0, 1, ..., nrpoc-1$
- **two_blocks**: Generates a two-block distribution, where the first and last block sizes are set to the specified average (avg), with all other blocks set to 0 for the specified number of processors (nproc). Optionally supports a many-to-many distribution (m2m). It follows that for each $$m_i = \begin{cases}\mathop{avg} && \text{if } i = 0 \text{ or } i = nproc-1 \\ 0 && \text{otherwise}\end{cases} $$ with $i = 0, 1, ..., nrpoc-1$


## Test suite

The test suite `test/suite.py` is designed to automate the process of running the benchmarking workflow described in [Usage](#usage). It allows the user to configure and execute a set of tests usind different MPI executors (`mpirun`, `srun`) and MPI implementations ( `OpenMPI`, `MPICH`). The test suite is highly configurable allowing for easy benchmarking under different scenarios. The results of each test case are saved in a user-friendly way. 

We provide test cases for each of the implemented collective operations and each of the message distributions in the `test/` folder. All of these files are also integrated into CMake (this takes a while!) and be run using

``` bash
cd build
cmake ..
make test
```

The `suite.py` file reads a JSON that defines the test cases to be run and creates either a `bash` script or a `slurm` job for each test case using the specified executor.

``` bash
python suite.py --help 
usage: suite.py [-h] [--ask] [--executor {mpirun,srun}] [--mpi-impl {openmpi,mpich}] [--wd WD]
                filename

positional arguments:
  filename

options:
  -h, --help            show this help message and exit
  --ask                 If set will ask for output directory name (default: False)
  --executor {mpirun,srun}
                        The job scheduler to use (default: mpirun)
  --mpi-impl {openmpi,mpich}
                        MPI implementation to use (default: openmpi)
  --wd WD               Working directory with binaries (default: .)
```

The test cases for `suite.py` file require a specific JSON format, which we enforce with [Pydantic](https://docs.pydantic.dev/latest/).

- `benchmark_name`: The name of the benchmark configuration.
- `test_suite`: A list of tests to execute. Each test has:

  - `test_name`: A unique identifier for the test. 
  - `test_type`: The type of measurement (e.g., throughput, latency). 
  - `collective`: The MPI collective program to run. 
  - `messages_data`: Either a filename or function parameters for the data. 
  - `timeout`: The timeout for the test (default is 1). 
- `global_config`: Defines global configurations like:
  - `max_runtime`: Maximum allowed runtime in seconds for all tests. 
  - `nproc`: Number of processes to run (optional). 
  - `output`: Defines output settings like where to save results and whether to display verbose output.


The `example.json` file serves as demonstration for how multiple test cases need to be defined

``` json
{
    "benchmark_name": "test_scatterv",
    "test_suite": [
        {
            "test_name": "scatterv-equal-100",
            "test_type": "latency",
            "collective": "scatterv",
            "messages_data": {
                "data": "equal",
                "params": {
                    "nproc": 4,
                    "val": 100
                }
            }
        },
        {
            "test_name": "scatterv-equal-10",
            "test_type": "latency",
            "collective": "scatterv",
            "messages_data": "custom.csv"
        },
        {
            "test_name": "scatterv-equal-100-32p",
            "test_type": "latency",
            "collective": "scatterv",
            "messages_data": {
                "data": "equal",
                "params": {
                    "nproc": 32,
                    "val": 100
                }
            }
        }
    ],
    "global_config": {
        "max_runtime": 3600,
        "nproc": 4,
        "output": {
            "directory": "./results",
            "verbose": true
        }
    }
}
```

which can be called using

``` 
python suite.py example.json
```

By default the above command will create three folders in the `results/` directory where for each test case a bash script will be writen in which the `scatterv` collective will be executed with four processes using `mpirun` as executor and the respective `messages_data` for the nubmer of messages that will be sent. The resulting latency output of `scatterv` will then be saved to the created directory. Notice that we have defined two cases: messages in a pre-defined CSV file and messages that are dynamically created and also saved to the output directory.  
