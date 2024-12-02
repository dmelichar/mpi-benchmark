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
// For dummy_compute
const int DIM = 10;

class IBcast {
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

    void dummy_compute() {
        double target_sec = 0.0;
        double time_elapsed = 0.0;
        

        while (time_elapsed < target_sec) {
            double t1 = MPI_Wtime();  
            
            // TODO

            double t2 = MPI_Wtime();
            time_elapsed += (t2-t1);
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

        MPI_Request request;
        MPI_Status status;

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

                double init_time = MPI_Wtime();
                MPI_Ibcast(buffer.data(), size, MPI_CHAR, 0, MPI_COMM_WORLD, &request);
                double init_time = MPI_Wtime() - init_time;

                tcomp = MPI_Wtime();
                // TODO Implement dummy_compute
                test_time = dummy_compute(latency_in_secs, &request);
                tcomp = MPI_Wtime() - tcomp;

                wait_time = MPI_Wtime();
                MPI_Wait(&request,&status);
                wait_time = MPI_Wtime() - wait_time;

                double t_stop = MPI_Wtime();

                // TODO Also need to add init_time, tcomp, wait_time to a vector
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

            // Calculate latency
            latency = (timer * 1e6) / iter;

            // Reduce operations to get min, max, and average times
            MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            avg_time /= csize;

            // TODO Also need to get init_time, tcomp, wait_time by reduce

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
        // TODO Need to add init_time, tcomb, wait_time
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
        IBcast benchmark;
        // Test different message sizes
        std::vector<size_t> msg_sizes = {1024};   
        for (auto msg_size : msg_sizes) {
            // min, max, seconds, verbose
            benchmark.run(1, msg_size, 1, true);
            benchmark.save_latencies("latencies.csv");
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

