# To-Do

The following features are up to be implemented

- [ ] Use JSON for test case definition
- [ ] Write statistics class
    - Mean, mode
    - Plot with e.g. [Matplot++](https://github.com/alandefreitas/matplotplusplus?tab=readme-ov-file)
    - Export to `csv`
- [ ] Benchmarking functions for
    - Collectives
        - [ ] `MPI_Allgather`
        - [ ] `MPI_Allgatherv`
        - [ ] `MPI_Alltoall`
        - [ ] `MPI_Alltoallv`
        - [ ] `MPI_Bcast`
        - [ ] `MPI_Gather`
        - [ ] `MPI_Gatherv`
        - [ ] `MPI_Reduce`
        - [ ] `MPI_Reduce_scatter`
        - [ ] `MPI_Scatter`
        - [ ] `MPI_Scatterv`
    - [ ] Non-blocking of the abov
    - One-Sided
        - [ ] `MPI_Put`
        - [ ] `MPI_Get` 
    - Send/Recive
        - [ ] `MPI_Recv`
        - [ ] `MPI_Send`
    - Reduction
        - [ ] `MPI_Allreduce
- [ ] Error handling
- [ ] Logging
- [ ] Max runtime
- [ ] Process check


Nur Rohdaten: Start Scatterv, endzeit beim letzten Prozess
Minimal dep: schema vllt, nur ein test case pro file
bound by time, not by iteration
gatherv von Zeiten auf allen Prozessen, aktuell nur auf root
Kein warmup
also test on other implementations of mpi



