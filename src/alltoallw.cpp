#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <iomanip>
#include <any>


#include <mpi.h>

class Alltoallw {
        int rank{};
        int csize{};
        std::vector<std::any> sbuffer;
        std::vector<std::any> rbuffer;
        std::vector<double> timings;
        std::vector<int> displs;
        std::vector<int> sendcounts;
        std::vector<int> recvcounts;
        std::vector<MPI_Datatype> sendtypes;

        // Prepare messages
        void setup(const std::string &filename)
        {
                sendcounts.resize(csize, 0);
                recvcounts.resize(csize, 0);
                displs.resize(csize, 0);
                sendtypes.resize(csize, MPI_INT);

                if (rank == 0) {
                        std::vector rows(csize, std::vector<int>(csize));
                        std::vector columns(csize, std::vector<int>(csize));

                        std::ifstream file(filename);
                        if (!file) {
                                std::cerr << "ERROR: Could not open file " << filename << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }

                        size_t prc = 0;
                        std::string line;
                        while (std::getline(file, line)) {
                                std::istringstream ss(line);
                                std::vector row(csize, 0);
                                std::string val;

                                size_t idx = 0;
                                while (std::getline(ss, val, ',')) {
                                        row[idx] = std::stoi(val);
                                        idx++;
                                        if (idx > csize) {
                                                // @formatter:off
                                                std::cerr << "ERROR: Number of columns "
                                                          << "(" << idx << ") "
                                                          << "does not match number of processes "
                                                          << "(" << csize << ")."
                                                          << std::endl;
                                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                                                // @formatter:on
                                        }
                                }

                                if (idx < csize) {
                                        // @formatter:off
                                        std::cerr << "ERROR: Number of columns "
                                                  << "(" << idx << ") "
                                                  << "does not match number of processes "
                                                  << "(" << csize << ")."
                                                  << std::endl;
                                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                                        // @formatter:on
                                }

                                rows[prc] = row;
                                for (size_t i = 0; i < csize; ++i) {
                                        columns[i][prc] = row[i];
                                }

                                prc++;
                                if (prc > csize) {
                                        std::cerr << "ERROR: Too many lines in file " << std::endl;
                                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                                }
                        }

                        if (prc != csize) {
                                std::cerr << "ERROR: Not enough lines in file " << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }

                        file.close();

                        for (int i = 0; i < csize; ++i) {
                                MPI_Send(rows[i].data(), csize, MPI_INT, i, 0, MPI_COMM_WORLD);
                                MPI_Send(columns[i].data(), csize, MPI_INT, i, 0, MPI_COMM_WORLD);
                        }

                } else {
                        MPI_Recv(sendcounts.data(), csize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(recvcounts.data(), csize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                for (int i = 0; i < csize; ++i) {
                        switch (i % 3) {
                        case 0:
                                sendtypes[i] = MPI_CHAR;
                                break;
                        case 1:
                                sendtypes[i] = MPI_INT;
                                break;
                        case 2:
                                sendtypes[i] = MPI_DOUBLE;
                                break;
                        default:
                                sendtypes[i] = MPI_INT;
                                break;
                        }
                }

                sbuffer.resize(std::accumulate(sendcounts.begin(), sendcounts.end(), 0));
                int value = 1, offset = 0;
                for (int i : sendcounts) {
                        std::fill_n(sbuffer.begin() + offset, i, value);
                        offset += i;
                        ++value;
                }

                rbuffer.resize(std::accumulate(recvcounts.begin(), recvcounts.end(), 0));

                for (int i = 1; i < csize; ++i) {
                        displs[i] = displs[i - 1] + sendcounts[i - 1];
                }

                MPI_Barrier(MPI_COMM_WORLD);
        }

public:
        Alltoallw()
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

                if (rank == 0 && verbose) {
                        // @formatter:off
                        std::cout << std::left
                                  << std::setw(25) << "Messages (count)"
                                  << std::setw(25) << "Avg Latency (s)"
                                  << std::setw(25) << "Min Latency (s)"
                                  << std::setw(25) << "Max Latency (s)"
                                  << std::endl;
                        // @formatter:on
                }

                // Global clock
                double global_start_time = 0.0;
                if (rank == 0) {
                        global_start_time = MPI_Wtime();
                }
                MPI_Bcast(&global_start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // TODO This could probably be improved
                while (true) {
                        const double t_start = MPI_Wtime();
                        MPI_Alltoallw(sbuffer.data(),
                                      sendcounts.data(),
                                      displs.data(),
                                      sendtypes.data(),
                                      rbuffer.data(),
                                      recvcounts.data(),
                                      displs.data(),
                                      sendtypes.data(),
                                      MPI_COMM_WORLD);
                        const double t_stop = MPI_Wtime();

                        timings.push_back(t_stop - t_start);
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

                const double min_local = *std::ranges::min_element(timings);
                const double max_local = *std::ranges::max_element(timings);
                const double avg_local = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();

                double min_time = 0.0, max_time = 0.0, avg_time = 0.0;
                MPI_Reduce(&min_local, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
                MPI_Reduce(&max_local, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&avg_local, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                avg_time /= csize;

                if (rank == 0 && verbose) {
                        // @formatter:off
                        std::cout << std::left
                                  << std::setw(25) << sbuffer.size()
                                  << std::setw(25) << avg_time * 1e6
                                  << std::setw(25) << min_time * 1e6
                                  << std::setw(25) << max_time * 1e6
                                  << std::endl
                                  << std::endl;
                        // @formatter:on
                }

                MPI_Barrier(MPI_COMM_WORLD);
                // @formatter:off
                std::ostringstream oss;
                oss << std::left
                    << std::setw(25) << "Rank " + std::to_string(rank)
                    << std::setw(25) << avg_local * 1e6
                    << std::setw(25) << min_local * 1e6
                    << std::setw(25) << max_local * 1e6;
                std::cout << oss.str() << std::endl;
                // @formatter:on

        }

        // Save data to file
        // TODO Needs improvement. Big file size too. If NFS, could use MPI_FILE
        void save_latencies(const std::string &filename, const bool verbose = false) const
        {
                const size_t num_timings = timings.size();
                std::vector<int> counts(csize);
                MPI_Gather(&num_timings, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

                // Process 0 creates or opens the file and writes the header and its own timings to the file first
                if (rank == 0) {
                        std::ofstream out_file(filename);
                        if (!out_file) {
                                std::cerr << "ERROR: Unable to open file " << filename << " for writing." << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }

                        out_file.seekp(0, std::ios::end);
                        if (out_file.tellp() == 0) {
                                out_file << "Rank,Iteration,Latency\n";
                        }
                        for (int i = 0; i < num_timings; ++i) {
                                out_file << rank << "," << i << "," << timings[i] << "\n";
                        }
                        out_file.close();
                } else {
                        MPI_Send(timings.data(), num_timings, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }

                // Process 0 receives and appends timings from all other processes
                if (rank == 0) {
                        std::ofstream out_file(filename, std::ios::app);
                        if (!out_file) {
                                std::cerr << "ERROR: Unable to open file " << filename << " for writing." << std::endl;
                                exit(EXIT_FAILURE);
                        }

                        std::vector<double> recv_timings;
                        for (int r = 1; r < csize; ++r) {
                                recv_timings.resize(counts[r]);
                                MPI_Recv(recv_timings.data(),
                                         counts[r],
                                         MPI_DOUBLE,
                                         r,
                                         0,
                                         MPI_COMM_WORLD,
                                         MPI_STATUS_IGNORE);
                                for (int i = 0; i < counts[r]; ++i) {
                                        out_file << r << "," << i << "," << recv_timings[i] << "\n";
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
        const option long_options[] = {
                        {"help", no_argument, nullptr, 'h'},
                        {"fmessages", required_argument, nullptr, 'm'},
                        {"foutput", required_argument, nullptr, 'o'},
                        {"timeout", required_argument, nullptr, 't'},
                        {"verbose", no_argument, nullptr, 'v'},
                        {nullptr, 0, nullptr, 0}
        };

        std::string fmessages = "default_messages.txt";
        std::string foutput = "default_output.txt";
        int timeout = 10;
        bool verbose = false;

        int opt;

        while ((opt = getopt_long(argc, argv, "hm:o:n:t:v", long_options, nullptr)) != -1) {
                switch (opt) {
                case 'h':
                        // @formatter:off
                        std::cout << "Help: This program runs a MPI alltoallw\n"
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
                Alltoallw benchmark;
                benchmark.run(fmessages, timeout, verbose);
                benchmark.save_latencies(foutput, verbose);
        } catch (const std::exception &e) {
                std::cerr << "ERROR: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
}