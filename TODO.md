# To-Do

The following features are up to be implemented

- [ ] Use JSON for test case definition: one test per file
- [ ] (optional) Write statistics class for mean, mode
- [ ] (optiona) Setup plots with e.g. [Matplot++](https://github.com/alandefreitas/matplotplusplus?tab=readme-ov-file)
- [x] Export response time and throughput to `csv`
- [ ] Benchmarking functions for
    - Collectives blocking
        - [ ] `MPI_Allgather`
        - [ ] `MPI_Allgatherv`
        - [ ] `MPI_Alltoall`
        - [ ] `MPI_Alltoallv`
        - [x] `MPI_Bcast`   (boilerplate for all)
        - [ ] `MPI_Gather`
        - [ ] `MPI_Gatherv`
        - [ ] `MPI_Reduce`
        - [ ] `MPI_Reduce_scatter`
        - [ ] `MPI_Scatter`
        - [ ] `MPI_Scatterv`
    - [ ] Collectives non-blocking
        - [ ] `MPI_IAllgather`
        - [ ] `MPI_IAllgatherv`
        - [ ] `MPI_IAlltoall`
        - [ ] `MPI_IAlltoallv`
        - [x] `MPI_IBcast` (needs testing)
        - [ ] `MPI_IGather`
        - [ ] `MPI_IGatherv`
        - [ ] `MPI_IReduce`
        - [ ] `MPI_IReduce_scatter`
        - [ ] `MPI_IScatter`
        - [ ] `MPI_IScatterv`
    - One-Sided
        - [ ] `MPI_Put`
        - [ ] `MPI_Get` 
    - Send/Recive
        - [ ] `MPI_Recv`
        - [ ] `MPI_Send`
    - Reduction
        - [ ] `MPI_Allreduce`
- [ ] Error handling in MPI 
- [ ] Logging with unique identifier
- [x] Setup bound by max runtime


