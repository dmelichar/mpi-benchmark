#!/bin/bash
#SBATCH --job-name=benchmark_mpi_job
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
spack load mpich@4.1.2
srun /home/staff/melichar/mpi-benchmark/test/../build/scatterv --fmessages /home/staff/melichar/mpi-benchmark/test/results/test-scatterv-alldist-32p-mpich/32-alternating.csv --foutput /home/staff/melichar/mpi-benchmark/test/results/test-scatterv-alldist-32p-mpich/scatterv-alternating.csv --timeout 1 --verbose 
