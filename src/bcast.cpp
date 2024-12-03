#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include <fstream>

#include <mpi.h>

// TODO Maybe other mod. Need though, otherwise filesize issues
const int TIMINGS_GRANULARITY = 100;

class Bcast {
private:
    int rank;
    int csize;
    std::vector<double> buffer;
    std::vector<double> timings;

    // Prepare messages 
    void setup (size_t max_msg_size) {
        try {
            buffer.resize(max_msg_size);
            // TODO Doesn't really makes a difference
            if (rank == 0) {
                std::iota(buffer.begin(), buffer.end(), 0); 
            } else {
                std::fill(buffer.begin(), buffer.end(), 0);
            }
        } catch (const std::bad_alloc& e) {
            std::cerr << "Could not allocate memory [rank " << rank << "]: " 
                      << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

public:
   Bcast () {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &csize);

        if (rank == 0) {
            if (csize < 2) {
                std::cerr << "ERROR: Need more than one process." << std::endl;
                MPI_Finalize();
                std::exit(EXIT_FAILURE);
            }
        }
    }

    void run (size_t min_msg_size, size_t max_msg_size, double max_seconds = 1, bool verbose = false) { 
        setup(max_msg_size);

        if (rank == 0 && verbose) {
            std::cout << std::left 
                      << std::setw(25) << "Size (Bytes)" 
                      << std::setw(25) << "Avg Latency (μs)" 
                      << std::setw(25) << "Min Latency (μs)" 
                      << std::setw(25) << "Max Latency (μs)" 
                      << std::endl;
        }
        
        // Incrementing messages
        for (size_t size = min_msg_size; size <= max_msg_size; size *= 2) {
            double timer = 0.0;
            double latency = 0.0;
            double min_time = 0.0, max_time = 0.0, avg_time = 0.0;
            int iter = 0;

            // Global clock
            double global_start_time = 0.0;
            if (rank == 0) {
                global_start_time = MPI_Wtime();
            }
            MPI_Bcast(&global_start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            // Time-based measurement
            while (true) {    
                double t_start = MPI_Wtime();
                MPI_Bcast(buffer.data(), size, MPI_CHAR, 0, MPI_COMM_WORLD);
                double t_stop = MPI_Wtime();

                timer += t_stop - t_start;
                iter++;
                if (iter % TIMINGS_GRANULARITY == 0) {
                    timings.push_back(timer);
                }
                MPI_Barrier(MPI_COMM_WORLD);

                bool continue_loop = true;
                if (rank == 0) {
                    double elapsed_time = MPI_Wtime() - global_start_time; 
                    continue_loop = (elapsed_time < max_seconds);
                }
                MPI_Bcast(&continue_loop, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
                if (!continue_loop) break;
            }
            MPI_Barrier(MPI_COMM_WORLD);

            // Calculate latency in microseconds
            latency = (timer * 1e6) / iter;

            // Reduce operations to get min, max, and average times
            MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            avg_time /= csize;

            if (rank == 0 && verbose) {
                std::cout << std::left
                          << std::setw(25) << size
                          << std::setw(25) << avg_time
                          << std::setw(25) << min_time
                          << std::setw(25) << max_time
                          << std::endl;
            }
        }
    }

    // Save data to file
    void save_latencies(const std::string &filename) {
        int num_timings = timings.size();
        std::vector<int> counts(csize);
        std::vector<int> displacements(csize);
        
        MPI_Gather(&num_timings, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displacements[0] = 0;
            for (int i = 1; i < csize; ++i) {
                displacements[i] = displacements[i - 1] + counts[i - 1];
            }
        }

        int total_timings = std::accumulate(counts.begin(), counts.end(), 0);
        std::vector<double> all_timings(total_timings);

        MPI_Gatherv(timings.data(),
                    num_timings,
                    MPI_DOUBLE, 
                    all_timings.data(),
                    counts.data(),
                    displacements.data(),
                    MPI_DOUBLE, 
                    0, 
                    MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::ofstream out_file(filename);
            if (!out_file) {
                std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
                exit(EXIT_FAILURE);
            }
            out_file << "Rank,Iteration,Latency\n";
            int idx = 0;
            for (int r = 0; r < csize; ++r) {
                for (int i = 0; i < counts[r]; ++i) {
                    out_file << r << "," << i << "," << all_timings[idx++] << "\n";
                }
            }
            out_file.close();
            std::cout << "Latencies saved to " << filename << std::endl;
        }

    }
};

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv); 
    try { 
        Bcast benchmark;
        // Test different message sizes
        std::vector<size_t> msg_sizes = {1024};   
        for (auto msg_size : msg_sizes) {
            // min, max, seconds, verbose
            benchmark.run(1, msg_size, 1, true);
            benchmark.save_latencies("bcast-latencies.csv");
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

