#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <mpi.h>

class Scatterv {

        int rank{};
        int csize{};
        std::vector<double> sbuffer;
        std::vector<double> rbuffer;
        std::vector<int> displs;
        std::vector<int> sendcounts;

        std::deque<double> starts;
        std::deque<double> ends;

        // Prepare messages
        void setup(const std::string &filename)
        {
                sendcounts.resize(csize, 0);
                displs.resize(csize, 0);

                if (rank == 0) {
                        std::ifstream file(filename);
                        if (!file) {
                                std::cerr << "ERROR: Could not open file " << filename << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }

                        std::string line;
                        if (!std::getline(file, line)) {
                                std::cerr << "ERROR: Could not read line " << filename << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }

                        std::istringstream ss(line);
                        std::vector<int> row;
                        std::string val;
                        while (std::getline(ss, val, ',')) {
                                row.push_back(std::stoi(val));
                        }
                        file.close();

                        if (row.size() != csize) {
                                // @formatter:off
                                std::cerr << "ERROR: Number of columns "
                                          << "(" << row.size() << ") "
                                          << "does not match number of processes "
                                          << "(" << csize << ")."
                                          << std::endl;
                                // @formatter:on
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }

                        sbuffer.resize(std::accumulate(row.begin(), row.end(), 0));
                        int value = 1, offset = 0;
                        for (int i : row) {
                                std::fill_n(sbuffer.begin() + offset, i, value);
                                offset += i;
                                ++value;
                        }

                        sendcounts = row;
                }

                MPI_Bcast(sendcounts.data(), csize, MPI_INT, 0, MPI_COMM_WORLD);
                rbuffer.resize(sendcounts[rank]);

                for (int i = 1; i < csize; ++i) {
                        displs[i] = displs[i - 1] + sendcounts[i - 1];
                }
        }

public:
        Scatterv()
        {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &csize);

                if (rank == 0 && csize < 2) {
                        std::cerr << "ERROR: Need more than one process." << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
        }

        void run(const std::string &filename, const double max_seconds = 1, const bool verbose = false)
        {
                setup(filename);

                // Global clock
                double global_start_time = 0.0;
                if (rank == 0) {
                        global_start_time = MPI_Wtime();
                }
                MPI_Bcast(&global_start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                MPI_Barrier(MPI_COMM_WORLD);
                while (true) {
                        const double t_start = MPI_Wtime();
                        MPI_Scatterv(sbuffer.data(),
                                     sendcounts.data(),
                                     displs.data(),
                                     MPI_DOUBLE,
                                     rbuffer.data(),
                                     sendcounts[rank],
                                     MPI_DOUBLE,
                                     0,
                                     MPI_COMM_WORLD);
                        const double t_stop = MPI_Wtime();

                        starts.push_back(t_start);
                        ends.push_back(t_stop);

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

                std::vector<int> call_starts(csize);
                std::vector<int> call_ends(csize);
                const int starts_size = static_cast<int>(starts.size());
                const int ends_size = static_cast<int>(ends.size());
                MPI_Gather(&starts_size, 1, MPI_INT, call_starts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Gather(&ends_size, 1, MPI_INT, call_ends.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (rank == 0) {
                        // @formatter:off
                        if (!std::ranges::all_of(call_starts.begin(), call_starts.end(), [&](const int x) {
                                    return x == call_starts[0];
                            })) {
                                std::cerr << "ERROR: Timing buffers mismatch: "
                                             "Process has different number of iterations in starts"
                                          << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }
                        if (!std::ranges::all_of(
                                call_ends.begin(), call_ends.end(), [&](const int x) { return x == call_ends[0]; })) {
                                std::cerr << "ERROR: Timing buffers mismatch: "
                                             "Process has different number of iterations in ends"
                                          << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }
                        if (call_starts.front() != call_ends.front()) {
                                std::cerr << "ERROR: Timing buffers mismatch: "
                                             "Starts and ends are different size"
                                          << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }
                        // @formatter:on
                }
                iter = static_cast<int>(starts.size());

                std::vector<double> lat(iter);
                for (int i = 0; i <= iter; ++i) {
                        lat[i] = ends[i] - starts[i];
                }

                std::vector<double> latencies;
                if (rank == 0) {
                        latencies.resize(iter * csize);
                }
                MPI_Gather(lat.data(), iter, MPI_DOUBLE, latencies.data(), starts_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                if (rank == 0 && verbose) {
                        // Compute local min, max, average for each p (i.e. for each process' time)
                        std::vector<double> min_local(csize);
                        std::vector<double> max_local(csize);
                        std::vector<double> avg_local(csize);
                        for (int i = 0; i < csize; ++i) {
                                std::vector<double> local(iter);
                                for (int j = 0; j < iter; ++j) {
                                        local[j] = latencies[j + i * iter];
                                }
                                min_local[i] = *std::ranges::min_element(local);
                                max_local[i] = *std::ranges::max_element(local);
                                avg_local[i] = std::accumulate(local.begin(), local.end(), 0.0) / iter;
                        }

                        // Output local

                        // @formatter:off
                        std::ostringstream oss1;
                        oss1 << std::left << std::setw(25) << ""
                                         << std::setw(25) << "Avg Latency (μs)"
                                         << std::setw(25) << "Min Latency (μs)"
                                         << std::setw(25) << "Max Latency (μs)"
                                         << std::endl;
                        for (int i = 0; i < csize; ++i) {
                                oss1 << std::left << std::setw(25) << "Rank " + std::to_string(i)
                                                 << std::setw(25) << avg_local[i] * 1e6
                                                 << std::setw(25) << min_local[i] * 1e6
                                                 << std::setw(25) << max_local[i] * 1e6
                                                 << std::endl;
                        }
                        std::cout << oss1.str() << std::endl;
                        // @formatter:on


                        // Compute global min, max, average by iteration over all processes
                        std::vector<double> min_global(iter);
                        std::vector<double> max_global(iter);
                        std::vector<double> avg_global(iter);
                        for (int i = 0; i < iter; ++i) {
                                std::vector<double> it(csize);
                                for (int j = 0; j < csize; ++j) {
                                        it[j] = latencies[i + j * csize];
                                }
                                min_global[i] = *std::ranges::min_element(it);
                                max_global[i] = *std::ranges::max_element(it);
                                avg_global[i] = std::accumulate(it.begin(), it.end(), 0.0) / csize;
                        }


                        double vmin_global = *std::ranges::min_element(min_global);
                        double vmax_global = *std::ranges::max_element(max_global);
                        double vavg_global = std::accumulate(avg_global.begin(), avg_global.end(), 0.0) / iter;

                        // @formatter:off
                        std::ostringstream oss2;
                        oss2 << std::left << std::setw(25) << "Global messages count"
                                         << std::setw(25) << "Avg Latency (μs)"
                                         << std::setw(25) << "Min Latency (μs)"
                                         << std::setw(25) << "Max Latency (μs)"
                                         << std::setw(25) << "Iterations"
                                         << std::endl
                                         << std::setw(25) << sbuffer.size()
                                         << std::setw(25) << vavg_global * 1e6
                                         << std::setw(25) << vmin_global * 1e6
                                         << std::setw(25) << vmax_global * 1e6
                                         << std::setw(25) << iter
                                         << std::endl
                                         << std::endl;
                        std::cout << oss2.str() << std::endl;
                        // @formatter:on

                }

                MPI_Barrier(MPI_COMM_WORLD);
        }

        // Save data to file
        void save_latencies(const std::string &filename, const bool verbose = false) const
        {
                if (starts.empty()  || ends.empty() || iter == -1) {
                        std::cerr << "ERROR: Must run first before saving" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }

                if (rank == 0) {
                        std::ofstream out_file(filename);
                        if (!out_file) {
                                std::cerr << "ERROR: Unable to open file " << filename << " for writing." << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }

                        out_file.seekp(0, std::ios::end);
                        if (out_file.tellp() == 0) {
                                out_file << "Rank,Iteration,Starttime,Endtime\n";
                        }
                        for (int i = 0; i < starts_size; ++i) {
                                out_file << rank << ","
                                                << i << ","
                                                << starts[i] << ","
                                                << ends[i] << "\n";
                        }
                        out_file.close();
                } else {
                        // Needs to be contiguous memory block
                        std::vector vec_starts(starts.begin(), starts.end());
                        std::vector vec_ends(ends.begin(), ends.end());
                        MPI_Send(vec_starts.data(), iter, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                        MPI_Send(vec_ends.data(), iter, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }

                if (rank == 0) {
                        std::ofstream out_file(filename, std::ios::app);
                        if (!out_file) {
                                std::cerr << "ERROR: Unable to open file " << filename << " for writing." << std::endl;
                                exit(EXIT_FAILURE);
                        }

                        std::vector<double> recv_starts(iter);
                        std::vector<double> recv_ends(iter);
                        for (int r = 1; r < csize; ++r) {
                                MPI_Recv(recv_starts.data(),
                                         iter,
                                         MPI_DOUBLE,
                                         r,
                                         0,
                                         MPI_COMM_WORLD,
                                         MPI_STATUS_IGNORE);
                                MPI_Recv(recv_ends.data(),
                                         iter,
                                         MPI_DOUBLE,
                                         r,
                                         0,
                                         MPI_COMM_WORLD,
                                         MPI_STATUS_IGNORE);
                                for (int i = 0; i < iter; ++i) {
                                        out_file << r << "," << i << "," << recv_starts[i] << "," << recv_ends[i] << "\n";
                                }
                        }
                        out_file.close();
                }

                if (rank == 0 && verbose) {
                        std::cout << "Latencies saved to " << filename << std::endl;
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
        int timeout = 10;
        bool verbose = false;

        int opt;

        while ((opt = getopt_long(argc, argv, "hm:o:n:t:v", long_options, nullptr)) != -1) {
                switch (opt) {
                case 'h':
                        // @formatter:off
                        std::cout << "Help: This program runs a MPI scatterv\n"
                                  << "Options:\n"
                                  << "  -h, --help            Show this help message\n"
                                  << "  -m, --fmessages FILE  Specify file with messages (default: default_messages.txt)\n"
                                  << "  -o, --foutput FILE    Specify output file (default: default_output.txt)\n"
                                  << "  -t, --timeout NUM     Specify timeout value in seconds (default: 10)\n"
                                  << "  -v, --verbose         Enable verbose mode\n";
                // @formatter:on
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
                Scatterv benchmark;
                benchmark.run(fmessages, timeout, verbose);
                benchmark.save_latencies(foutput, verbose);
        } catch (const std::exception &e) {
                std::cerr << "ERROR: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
}