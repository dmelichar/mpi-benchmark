#!/bin/bash
#SBATCH --job-name=benchmark_mpi_job
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
spack load openmpi@4.1.6
srun /home/staff/melichar/mpi-benchmark/test/../build/scatterv --fmessages /home/staff/melichar/mpi-benchmark/test/results/test-scatterv-alldist-32p-openmpi/32-normal.csv --foutput /home/staff/melichar/mpi-benchmark/test/results/test-scatterv-alldist-32p-openmpi/scatterv-normal.csv --timeout 1 --verbose 
