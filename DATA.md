# Data generating process

The data generating process (dgp) is used for the messages that are being sent by the MPI collective calls. There are three types collectives reflecting the message buffer.

- Fixed size: `Alltoall`, `Scatter`, `Gather`, ...
- Variable size: `Alltoallv`, `Scatterv`, `Gatherv`, ...
- Different dtype: `Alltoallw`, ...

We can also differentiate by the communication pattern.

- One-to-one: `Scatter`, `Gather`, `Scatterv`, `Gatherv`, ...
- Many-to-one: `Alltoall`, `Allgather`, ...

## Fixed size

For fixed size `Scatter` we will have something like and assuming we have four processes where for each `my_value` is its buffer

    int my_value;
        int buffer[4] = {0, 100, 200, 300};

    // For root
        MPI_Scatter(buffer, 1, MPI_INT, &my_value, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    // For non-root
        MPI_Scatter(NULL, 1, MPI_INT, &my_value, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

Similarly for `Gather` and again assuming four processes we will have something like the below. However, the value of each process' buffer should be different.

    // Process 1
        int my_value = 10;
    // Process 2
        int my_value = 20;
        // Process 3
        int my_value = 30;
        // Process 4
        int my_value = 40;

    // For root
        int buffer[4];
        MPI_Gather(&my_value, 1, MPI_INT, buffer, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

    // For non-root
        MPI_Gather(&my_value, 1, MPI_INT, NULL, 0, MPI_INT, root_rank, MPI_COMM_WORLD);

## Many-to-many

So far we have only considered the so called one-to-many calls that are essentially `Send` by one process and `Recive` by many processes in case of `Scatter` and in the case of `Gather` are essentially `Send` by many processes and `Recive` by one process. In many-to-many calls multiple processes may perform a `Send` and multuple will need ot `Recive`. For example, the `Allgather` assume that there are multiple processes with a unique buffer (just like in `Gather`) but in this case multiple buffers recive the buffers' content instead of only one process. This can be thought of as a `Gather` to a process followed by a `Bcast` from that process.

    // Process 1
        int my_value = 10;
    // Process 2
        int my_value = 20;
        // Process 3
        int my_value = 30;

    int buffer[3]
        MPI_Allgather(&my_value, 1, MPI_INT, buffer, 1, MPI_INT, MPI_COMM_WORLD);

In an `Alltoall` we essentially combine `Scatter` and `Gather`. Each process sends specific elements from its buffer to all other processes. It is similar to an `Allgather` but each process has its own buffer and sends elements from its buffer to others.

    // Process 1
        int my_value = {10, 20, 30};
    // Process 2
        int my_value = {100, 200, 300};
        // Process 3
        int my_value = {1000, 2000, 3000};

    int buffer[3]
        MPI_Alltoall(&my_values, 1, MPI_INT, buffer, 1, MPI_INT, MPI_COMM_WORLD);

    // Afterwards process 1: buffer = {10, 100, 1000}

## Variable size

For variable size we will have something like. The `counts` parameter states how many elements will be sent/recived and the `displacement` parameter states where they are. For example, assuming we have three processes $p1, p2, p3$, we define counts as `[1,2,1]` and displacements as `[0,2,6]` then the following buffer

    int buffer[7] = {100, 0, 101, 102, 0, 0, 103};

gives that $p1$ will have `100`, $p2$ will have `101,102` and $p3$ will have `103`.

## Different data type

Take a look [here](https://rookiehpc.org/mpi/docs/mpi_alltoallw/index.html). For simplicity, the dpg should just support `char`, `int` and `double`.

## Function interface

We can think of the function call

    def dgp(xdim, ydim, dtype, dist, counts, seed):
        """
            dtype in []
        """
        pass
