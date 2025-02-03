# This is basically a fancy mpirun wrapper.
# Alternatively, one could have simple shell scripts 
# that would be the _testcases_ or in other words
# the different benchmarks to run

import pathlib
import argparse
import csv

import numpy as np

from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field, ValidationError


class CommunicationPattern(BaseModel):
    mode: str = Field( 
        description="Blocking or non-blocking communication mode", 
        enum=["blocking", "non_blocking"]
    )
    collective_operation: str = Field(
        description="Type of collective operation if applicable",
        enum=[
            "bcast",
            "scatter",
            "scatterv",
            "gather",
            "gatherv",
            "allgather",
            "allgatherv",
            "alltoall",
            "alltoallv"
        ],
    )

class TestSuiteItem(BaseModel):
    test_name: str = Field( 
        description="Identifier for the specific test"
    )
    test_type: str = Field(
        description="Type of measurement to perform",
        enum=["latency", "bandwidth", "message_rate"]
    )
    communication_pattern: CommunicationPattern
    messages_data: str = Field(
        description="Filename of messages from data.py"
    )


class GlobalConfigOutput(BaseModel):
    encoding: str = Field(
        "csv",
        description="Output format",
        enum=["csv", "json", "text"]
    )
    directory: str = Field(
        "./results",
        description="Directory to save results and parameters to"
    )
    verbose: bool = Field(
        False,
        description="Verbose output"
    )

class GlobalConfig(BaseModel):
    max_runtime: int = Field(
        ge=0,
        description="Maximum runtime in seconds for each test"
    )
    processes: int = Field(
        ge=2,
        description="Number of processes"
    )
    output: GlobalConfigOutput = None


class OpenMPIBenchmarkConfig(BaseModel):
    benchmark_name: str = Field(description="Name of the benchmark configuration")
    test_suite: List[TestSuiteItem]
    global_config: GlobalConfig

    class Config:
        str_min_length = 1  # Ensures strings are not empty
        str_strip_whitespace = True  # Trims whitespace from strings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    
    try:
        json_string = pathlib.Path(args.filename).read_text()
        benchmark = OpenMPIBenchmarkConfig.model_validate_json(json_string, strict=True)
        # setup output
        # for each test:
        #   call cpp file
        #   ex: mpirun -np 4 bcast -f 4-normal.csv -o bcast-4-normal-latencies -trials 10 --timeout 10 --verbose         
    except ValidationError as e:
        print(e)
        raise SystemExit(1)
    except Exception as e:
        print(e)
        raise SystemExit(1)

if __name__ == "__main__":
    main()