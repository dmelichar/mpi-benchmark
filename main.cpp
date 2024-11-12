#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>
#include <fstream>

class MPIBenchmark {
  private:
    int rank;
    int size;
    std::vector<double> timings;

    // Helper function to calculate statistics
    void calculate_statistics(
            const std::vector<double> &measurements,
            double &min,
            double &max,
            double &avg,
            double &stddev
    ) {
        min = *std::min_element(measurements.begin(), measurements.end());
        max = *std::max_element(measurements.begin(), measurements.end());
        //  T accumulate( InputIt first, InputIt last, T init );
        //  Computes the sum of the given value init and the elements in the range [first, last). 
        avg = std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
       
        // x - avg
        std::vector<double> diff(measurements.size());
        std::transform(measurements.begin(), measurements.end(), diff.begin(), [avg](double x) { return x - avg; } );
        // T inner_product( InputIt1 first1, InputIt1 last1, InputIt2 first2, T init );
        // x*x
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0 );
        // s_x = sqrt(||x-\hat{x}|| / n)
        stddev = std::sqrt(sq_sum / measurements.size());
    }

  public:
    MPIBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size < 2) {
            std::cout << "ERROR: Need more than one process." << std::endl;
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    void benchmark_scatterv(
            size_t base_msg_size, 
            int iterations,
            const std::string &distribution = "increasing"
    ) {
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
            std::iota(sendbuf.begin(), sendbuf.end(), 0); // Fill with sequential numbers
        }
        
        // Warmup iterations. 
        // TODO Why?
        for (int i = 0; i < 100; i++) {
            MPI_Scatterv(
                    sendbuf.data(),
                    sendcounts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    recvbuf.data(),
                    sendcounts[rank],
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD
            );
        }

        for (int i = 0; i < iterations; i++) {
            // TODO Why?
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
            double min, max, avg, stddev;
            calculate_statistics(timings, min, max, avg, stddev);

            // Calculate bandwidth
            double total_bytes = total_elements * sizeof(double);
            double bandwidth = (total_bytes / 1024.0 / 1024.0) / (avg / 1e6); // MB/s

            std::cout << "\nScatterv Results (Distribution: " << distribution
                      << ")\n"
                      << "Message size: " << base_msg_size << " bytes\n"
                      << "Total elements: " << total_elements << "\n"
                      << "Latency (microseconds):\n"
                      << "  Min: " << min << "\n"
                      << "  Max: " << max << "\n"
                      << "  Avg: " << avg << "\n"
                      << "  StdDev: " << stddev << "\n"
                      << "Bandwidth: " << bandwidth << " MB/s\n"
                      << std::endl;
        }
    }
};

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv); 
    MPIBenchmark benchmark;

    // Test different message sizes
    std::vector<size_t> msg_sizes = {1024, 2048, 4096, 8192};    
    for (auto msg_size : msg_sizes) {
        // Run benchmark with increasing distribution
        benchmark.benchmark_scatterv(msg_size, 1000, "increasing");
        benchmark.benchmark_scatterv(msg_size, 1000, "uniform");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

