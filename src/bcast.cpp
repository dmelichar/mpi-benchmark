#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <iomanip>

#include <mpi.h>

// TODO Maybe other mod. Needed though, otherwise filesize issues
constexpr int TIMINGS_GRANULARITY = 100;

class Bcast {
        int rank{};
        int csize{};
        std::vector<double> buffer;
        std::vector<double> timings;

        // Prepare messages
        void setup(const size_t msg_size)
        {
                try {
                        buffer.resize(msg_size);
                        std::ranges::fill(buffer, 0);
                } catch (const std::bad_alloc &e) {
                        std::cerr << "Could not allocate memory [rank " << rank << "]: " << e.what() << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
        }

public:
        Bcast()
        {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &csize);

                if (rank == 0 && csize < 2) {
                        std::cerr << "ERROR: Need more than one process." << std::endl;
                        MPI_Finalize();
                        std::exit(EXIT_FAILURE);
                }
        }

        void run(const size_t msg_size, const double max_seconds = 1, const bool verbose = false)
        {
                setup(msg_size);
                int msg_size_int;
                if (msg_size <= static_cast<size_t>(std::numeric_limits<int>::max())) {
                        msg_size_int = static_cast<int>(msg_size);
                } else {
                        msg_size_int = std::numeric_limits<int>::max();
                }

                if (rank == 0 && verbose) {
                        // clang-format off
                        std::cout << std::left
                                        << std::setw(25) << "Size (Bytes)"
                                        << std::setw(25) << "Avg Latency (μs)"
                                        << std::setw(25) << "Min Latency (μs)"
                                        << std::setw(25) << "Max Latency (μs)"
                                        << std::endl;
                        // clang-format on
                }

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
                        const double t_start = MPI_Wtime();
                        MPI_Bcast(buffer.data(), msg_size_int, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        const double t_stop = MPI_Wtime();

                        timer += t_stop - t_start;
                        iter++;
                        if (iter % TIMINGS_GRANULARITY == 0) {
                                timings.push_back(timer);
                        }
                        MPI_Barrier(MPI_COMM_WORLD);

                        bool continue_loop = true;
                        if (rank == 0) {
                                const double elapsed_time = MPI_Wtime() - global_start_time;
                                continue_loop = elapsed_time < max_seconds;
                        }
                        MPI_Bcast(&continue_loop, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
                        if (!continue_loop)
                                break;
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
                        // clang-format off
                        std::cout << std::left
                                        << std::setw(25) << msg_size
                                        << std::setw(25) << avg_time
                                        << std::setw(25) << min_time
                                        << std::setw(25) << max_time
                                        << std::endl;
                        // clang-format on
                }

        }

        // Save data to file
        void save_latencies(const std::string &filename, const bool verbose = false) const
        {
                int num_timings;
                if (timings.size() <= static_cast<size_t>(std::numeric_limits<int>::max())) {
                        num_timings = static_cast<int>(timings.size());
                } else {
                        num_timings = std::numeric_limits<int>::max();
                }

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
                        // TODO: Alternative formats to csv
                        out_file << "Rank,Iteration,Latency\n";
                        int idx = 0;
                        for (int r = 0; r < csize; ++r) {
                                for (int i = 0; i < counts[r]; ++i) {
                                        out_file << r << "," << i << "," << all_timings[idx++] << "\n";
                                }
                        }
                        out_file.close();
                        if (verbose) {
                                std::cout << "Latencies saved to " << filename << std::endl;
                        }
                }
        }
};

int main(int argc, char *argv[])
{
        const option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                       {"fmessages", required_argument, nullptr, 'm'},
                                       {"foutput", required_argument, nullptr, 'o'},
                                       {"timeout", required_argument, nullptr, 't'},
                                       {"verbose", no_argument, nullptr, 'v'},
                                       {nullptr, 0, nullptr, 0}};

        std::string foutput = "default_output.txt";
        int timeout = 10;
        bool verbose = false;
        int opt;

        while ((opt = getopt_long(argc, argv, "hm:o:n:t:v", long_options, nullptr)) != -1) {
                switch (opt) {
                case 'h':
                        // clang-format off
                        std::cout << "Help: This program runs a MPI bcast\n"
                                  << "Options:\n"
                                  << "  -h, --help            Show this help message\n"
                                  << "  -m, --fmessages FILE  Specify file with messages (default: default_messages.txt)\n"
                                  << "  -o, --foutput FILE    Specify output file (default: default_output.txt)\n"
                                  << "  -t, --timeout NUM     Specify timeout value in seconds (default: 10)\n"
                                  << "  -v, --verbose         Enable verbose mode\n";
                        // clang-format on
                        return EXIT_SUCCESS;
                case 'm':
                        // Not implemented
                        break;
                case 'o':
                        foutput = optarg;
                        break;
                case 'v':
                        verbose = true;
                        break;
                case 't':
                        timeout = std::stoi(optarg);
                        break;
                default:
                        std::cerr << "Unknown option\n";
                        std::cerr << optarg << std::endl;
                        return EXIT_FAILURE;
                }
        }

        MPI_Init(&argc, &argv);
        try {
                Bcast benchmark;
                benchmark.run(1024, timeout, verbose);
                benchmark.save_latencies(foutput, verbose);
        } catch (const std::exception &e) {
                std::cerr << "Error: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
}