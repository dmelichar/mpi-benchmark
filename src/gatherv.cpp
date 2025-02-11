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

// TODO Maybe other mod. Need though, otherwise filesize issues
const int TIMINGS_GRANULARITY = 100;

class Gatherv {

        int rank{};
        int csize{};
        std::vector<double> sbuffer;
        std::vector<double> rbuffer;
        std::vector<double> timings;

        // Prepare messages
        void setup(const std::string &filename)
        {
                // TODO
        }

      public:
        Gatherv()
        {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &csize);

                if (rank == 0 && csize < 2) {
                        std::cerr << "ERROR: Need more than one process." << std::endl;
                        MPI_Finalize();
                        std::exit(EXIT_FAILURE);
                }
        }

        void run(const std::string &filename, const double max_seconds = 1, const bool verbose = false)
        {
                setup(filename);

                if (rank == 0 && verbose) {
                        // clang-format off
                        std::cout << std::left << std::setw(25) << "Size (Bytes)"
                                               << std::setw(25) << "Avg Latency (μs)"
                                               << std::setw(25) << "Min Latency (μs)"
                                               << std::setw(25) << "Max Latency (μs)"
                                  << std::endl;
                        // clang-format on
                }

                double timer = 0.0;
                double latency = 0.0;
                double min_time = 0.0, max_time = 0.0, avg_time = 0.0;

                // Global clock
                double global_start_time = 0.0;
                if (rank == 0) {
                        global_start_time = MPI_Wtime();
                }
                MPI_Bcast(&global_start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                int iter = 0;
                // Time-based measurement
                while (true) {
                        double t_start = MPI_Wtime();

                        // MPI_Gatherv(sbuffer.data(),, MPI_DOUBLE, rbuffer.data(), size, MPI_DOUBLE,
                        // MPI_COMM_WORLD);
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
                                  //<< std::setw(25) << buffer.size()
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

        std::string fmessages = "default_messages.txt";
        std::string foutput = "default_output.txt";
        int timeout = 30;
        bool verbose = false;

        int opt;

        while ((opt = getopt_long(argc, argv, "hm:o:n:t:v", long_options, nullptr)) != -1) {
                switch (opt) {
                case 'h':
                        std::cout
                            << "Help: This program runs a MPI bcast .\n"
                            << "Options:\n"
                            << "  -h, --help            Show this help message\n"
                            << "  -m, --fmessages FILE  Specify file with messages (default: default_messages.txt)\n"
                            << "  -o, --foutput FILE    Specify output file (default: default_output.txt)\n"
                            << "  -t, --timeout NUM     Specify timeout value in seconds (default: 30)\n"
                            << "  -v, --verbose         Enable verbose mode\n";
                        return EXIT_SUCCESS;
                case 'm':
                        fmessages = optarg;
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
                Gatherv benchmark;
                benchmark.run(fmessages, timeout, verbose);
                benchmark.save_latencies(foutput, verbose);
        } catch (const std::exception &e) {
                std::cerr << "Error: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
}
