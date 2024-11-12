#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>
#include <fstream>

#include <nlohmann/json-schema.hpp>

using nlohmann::json;
using nlohmann::json_schema::json_validator;

class MPIBenchmark {
  private:
    int rank;
    int size;
    std::vector<double> timings;

    // Helper function to calculate statistics
    double calculate_statistics(
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
        
        // T inner_product( InputIt1 first1, InputIt1 last1, InputIt2 first2, T init );
        double sq_sum = std::inner_product(
                measurements.begin(), 
                measurements.end(), 
                measurements.begin(), 
                0.0
        );
        stddev = std::sqrt(sq_sum / measurements.size() - avg * avg);
        return avg;
    }

  public:
    MPIBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    void benchmark_scatterv(
            size_t base_msg_size, 
            int iterations,
            const std::string &distribution = "increasing"
    ) {
        // Prepare send counts and displacements
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size);

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
        if (rank == 0) {
            sendbuf.resize(total_elements);
            std::iota(sendbuf.begin(), sendbuf.end(), 0); // Fill with sequential numbers
        }

        std::vector<double> recvbuf(sendcounts[rank]);

        // Warmup iterations
        for (int i = 0; i < 100; i++) {
            //    int MPI_Scatterv(const void* buffer_send,
            //         const int counts_send[],
            //         const int displacements[],
            //         MPI_Datatype datatype_send,
            //         void* buffer_recv,
            //         int count_recv,
            //         MPI_Datatype datatype_recv,
            //         int root,
            //         MPI_Comm communicator);
            MPI_Scatterv(sendbuf.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, recvbuf.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Actual measurement
        timings.clear();
        timings.reserve(iterations);

        for (int i = 0; i < iterations; i++) {
            MPI_Barrier(MPI_COMM_WORLD); // Synchronize before measurement

            auto start = std::chrono::high_resolution_clock::now();

            MPI_Scatterv(sendbuf.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, recvbuf.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
    json schema_json, example_json;

    std::ifstream schema_file("schema.json");
    if (!schema_file.is_open()) {
        std::cerr  << "Error opening schema file.\n";
    }
    schema_file >> schema_json;

    json_validator validator;
    validator.set_root_schema(schema_json);

    std::ifstream data_file("example.json");
    if (!data_file.is_open()) {
        std::cerr << "Error opening data file.\n";
        return 1;
    }
    data_file >> example_json;

    try {
        validator.validate(example_json);
        std::cout << "JSON is valid.\n";
    } catch (const std::exception &e) {
        std::cerr << "Validation error: " << e.what() << "\n";
    }


    MPI_Init(&argc, &argv);

    MPIBenchmark benchmark;

    // Test different message sizes
    std::vector<size_t> msg_sizes = {1024, 4096, 16384, 65536, 262144, 1048576};

    for (auto msg_size : msg_sizes) {
        // Run benchmark with increasing distribution
        benchmark.benchmark_scatterv(msg_size, 1000, "increasing");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

