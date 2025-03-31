# MPI Benchmark

This document describes the 

## Requirements

The benchmark was tested with the following MPI implementations on a Slrum cluster with 32 nodes.

- OpenMPI 4.1.6
- MPICH 4.1.2

For the test suite at least Python 3.9 is required.



In this document I describe the following:

- the implementation of the MPI collective call benchmark
    - CLI parameters
    - message input encoding
    - output encoding
    - open tasks: alltoallw
- the implementation of the test suite
    - CLI parameters
    - Requirements and Python version
    - JSON Schema
    - Result plotting
    - Compression
    - open tasks: ?