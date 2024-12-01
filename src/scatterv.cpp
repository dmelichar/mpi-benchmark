#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>
#include <fstream>

class Scatterv {
  private:
    int rank;
    int size;
    std::vector<double> timings;
    
  public:
    Scatterv() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size < 2) {
            std::cout << "ERROR: Need more than one process." << std::endl;
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }
    
    // Prepare messages according to distr
    void setup () {

    }

    // Save data to file
    void save() {

    }

    void run (int iterations) {
        // For calculating timings
        double timer = 0.0;
        double latency = 0.0;
        double t_start = 0.0;
        double t_stop = 0.0;

        // Prepare send counts and displacements
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size);

        // Prepare measurement vector
        timings.clear();
        timings.reserve(iterations);
    
        // Calculate send counts based on distribution pattern
        size_t total_elements = 0;
        for (int i = 0; i < size; i++) {
            if (distribution == "increasing") {
                sendcounts[i] = base_msg_size * (i + 1) / sizeof(double);
            } else if (distribution == "uniform") {
                sendcounts[i] = base_msg_size / sizeof(double);
            }
            displs[i] = total_elements;
            total_elements += sendcounts[i];
        }

        // Prepare send and receive buffers
        std::vector<double> sendbuf;
        std::vector<double> recvbuf(sendcounts[rank]);
        if (rank == 0) {
            sendbuf.resize(total_elements);
            std::iota(sendbuf.begin(), sendbuf.end(), 0); 
        }

        for (int i = 0; i < iterations; i++) {
            MPI_Barrier(MPI_COMM_WORLD); 

            auto start = std::chrono::high_resolution_clock::now();
            MPI_Scatterv(
                    sendbuf.data(),         // dispatch this from root p
                    sendcounts.data(),      // number of elements to send to each p
                    displs.data(),          // stride between data in elem
                    MPI_DOUBLE,             // send data type
                    recvbuf.data(),         // store messages here
                    sendcounts[rank],       // number of elem per p
                    MPI_DOUBLE,             // recive data type
                    0,                      // root p
                    MPI_COMM_WORLD          // comm
            );
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            timings.push_back(duration);
        }

        // Calculate statistics
        if (rank == 0) {
        }
    }
};

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv); 
    Scatterv benchmark;

    // Test different message sizes
    std::vector<size_t> msg_sizes = {1024, 2048, 4096, 8192};    
    for (auto msg_size : msg_sizes) {
        // Run benchmark with increasing distribution
        benchmark.run(msg_size, 1000, "increasing");
        benchmark.run(msg_size, 1000, "uniform");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

