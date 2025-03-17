#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <string>

#include <mpi.h>

template <typename T>
class Scatterv {

        int rank;
        int csize;

        T *sbuffer;
        T *rbuffer;

        int *displs;
        int *sendcounts;

        std::deque<double> times {};

        static MPI_Datatype get_mpi_type() {
                if constexpr (std::is_same_v<T, int>) {
                        return MPI_INT;
                } else if constexpr (std::is_same_v<T, double>) {
                        return MPI_DOUBLE;
                } else if constexpr (std::is_same_v<T, char>) {
                        return MPI_CHAR;
                }
                return MPI_DATATYPE_NULL;
        }

public:

        explicit Scatterv(const std::string &filename)
        {
                rank = -1;
                csize = -1;

                sbuffer = nullptr;
                rbuffer = nullptr;
                displs = nullptr;
                sendcounts = nullptr;

                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &csize);

                if (rank == 0 && csize < 2) {
                        std::cerr << "ERROR: Need more than one process." << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }

                sendcounts = new int[csize];
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
                                          << "(" << csize << ")." << std::endl;
                                // @formatter:on
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }


                        int msg_size = std::accumulate(row.begin(), row.end(), 0);
                        sbuffer = new T[msg_size];
                        int value = 1, offset = 0;
                        for (int i : row) {
                                std::fill_n(sbuffer + offset, i, static_cast<T>(value));
                                offset += i;
                                ++value;
                        }

                        for (int i = 0; i < row.size(); ++i) {
                                sendcounts[i] = row[i];
                        }
                }

                MPI_Bcast(sendcounts, csize, MPI_INT, 0, MPI_COMM_WORLD);
                rbuffer = new T[sendcounts[rank]];

                displs = new int[csize];
                displs[0] = 0;
                for (int i = 1; i < csize; ++i) {
                        displs[i] = displs[i - 1] + sendcounts[i - 1];
                }
        }

        ~Scatterv() {
                delete[] sbuffer;
                delete[] rbuffer;
                delete displs;
                delete sendcounts;
        }

        void run(const double max_seconds = 1, const bool verbose = false)
        {
                // Global clock
                double global_start_time = 0.0;
                if (rank == 0) {
                        global_start_time = MPI_Wtime();
                }
                MPI_Bcast(&global_start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                MPI_Barrier(MPI_COMM_WORLD);
                while (true) {
                        const double t_start = MPI_Wtime();
                        MPI_Scatterv(sbuffer,
                                     sendcounts,
                                     displs,
                                     get_mpi_type(),
                                     rbuffer,
                                     sendcounts[rank],
                                     get_mpi_type(),
                                     0,
                                     MPI_COMM_WORLD);
                        const double t_stop = MPI_Wtime();

                        times.push_back(t_start);
                        times.push_back(t_stop);

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

                const int iter = static_cast<int>(times.size()) / 2;

                std::vector<int> call_times(csize);
                MPI_Gather(&iter, 1, MPI_INT, call_times.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

                if (rank == 0) {
                        // @formatter:off
                        if (!std::ranges::all_of(
                                call_times.begin(),
                                call_times.end(),
                                [&](const int x)
                                {
                                    return x == call_times[0];
                                })) {
                                std::cerr << "ERROR: Timing buffers mismatch: "
                                             "Process has different number of iterations in starts"
                                          << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }
                        // @formatter:on
                }

                std::vector<double> lat(iter);
                for (int i = 0; i <= iter; ++i) {
                        lat[i] = times[i+1] - times[i];
                }

                double min_local = *std::ranges::min_element(lat);
                double max_local = *std::ranges::max_element(lat);
                double avg_local = std::accumulate(lat.begin(), lat.end(), 0.0) / iter;

                std::vector<double> min_locals(csize);
                std::vector<double> max_locals(csize);
                std::vector<double> avg_locals(csize);

                // Perform the reduction to gather global min, max, and average
                MPI_Gather(&min_local, 1, MPI_DOUBLE, min_locals.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gather(&max_local, 1, MPI_DOUBLE, max_locals.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gather(&avg_local, 1, MPI_DOUBLE, avg_locals.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


                if (rank == 0 && verbose) {
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
                                                 << std::setw(25) << avg_locals[i] * 1e6
                                                 << std::setw(25) << min_locals[i] * 1e6
                                                 << std::setw(25) << max_locals[i] * 1e6
                                                 << std::endl;
                        }
                        std::cout << oss1.str() << std::endl;
                        // @formatter:on


                        // Compute global min, max, average by iteration over all processes
                        double min_global = *std::ranges::min_element(min_locals);
                        double max_global = *std::ranges::max_element(max_locals);
                        double avg_global = std::accumulate(avg_locals.begin(), avg_locals.end(), 0.0) / csize;


                        int msg_size = 0;
                        for (int i = 0; i < csize; ++i) {
                                msg_size += sendcounts[i];
                        }
                        // @formatter:off
                        std::ostringstream oss2;
                        oss2 << std::left << std::setw(25) << "Global messages count"
                                         << std::setw(25) << "Avg Latency (μs)"
                                         << std::setw(25) << "Min Latency (μs)"
                                         << std::setw(25) << "Max Latency (μs)"
                                         << std::setw(25) << "Iterations"
                                         << std::endl
                                         << std::setw(25) << msg_size
                                         << std::setw(25) << avg_global * 1e6
                                         << std::setw(25) << min_global * 1e6
                                         << std::setw(25) << max_global * 1e6
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
                if (times.empty()) {
                        std::cerr << "ERROR: Must run first before saving" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }

                const int iter = static_cast<int>(times.size()) / 2;

                if (rank == 0) {
                        std::ofstream out_file(filename);
                        if (!out_file) {
                                std::cerr << "ERROR: Unable to open file " << filename << " for writing." << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }

                        out_file.seekp(0, std::ios::end);
                        if (out_file.tellp() == 0) {
                                out_file << "Rank,Iteration,Starttime,Endtime\n";
                                out_file.flush();
                        }
                        for (int i = 0; i < iter; ++i) {
                                out_file << rank << ","
                                         << i << ","
                                         << std::fixed << std::setprecision(8) << times[i] << ","
                                         << std::fixed << std::setprecision(8) << times[i+1] << "\n";
                        }
                        out_file.close();
                } else {
                        // Needs to be contiguous memory block
                        std::vector<double> vec_times(times.size());
                        for (int i = 0; i < times.size(); ++i) {
                                vec_times[i] = times[i];
                        }
                        MPI_Send(vec_times.data(), times.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }

                if (rank == 0) {
                        std::ofstream out_file(filename, std::ios::app);
                        if (!out_file) {
                                std::cerr << "ERROR: Unable to open file " << filename << " for writing." << std::endl;
                                exit(EXIT_FAILURE);
                        }

                        for (int r = 1; r < csize; ++r) {
                                std::vector<double> recv_vec_times(times.size());
                                MPI_Recv(recv_vec_times.data(),
                                         times.size(),
                                         MPI_DOUBLE,
                                         r,
                                         0,
                                         MPI_COMM_WORLD,
                                         MPI_STATUS_IGNORE);

                                for (int i = 0; i < iter; ++i) {
                                        out_file << r << ","
                                                 << i << ","
                                                 << std::fixed << std::setprecision(8) << recv_vec_times[i] << ","
                                                 << std::fixed << std::setprecision(8) << recv_vec_times[i+1] << "\n";
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
                                        {"dtype", required_argument, nullptr, 'd'},
                                       {nullptr, 0, nullptr, 0}};

        std::string fmessages = "default_messages.txt";
        std::string foutput = "default_output.txt";
        int timeout = 10;
        bool verbose = false;
        std::string dtype = "double";


        int opt;

        while ((opt = getopt_long(argc, argv, "hm:o:n:t:d:v", long_options, nullptr)) != -1) {
                switch (opt) {
                case 'h':
                        // @formatter:off
                        std::cout << "Help: This program runs a MPI scatterv\n"
                                  << "Options:\n"
                                  << "  -h, --help            Show this help message\n"
                                  << "  -m, --fmessages FILE  Specify file with messages (default: default_messages.txt)\n"
                                  << "  -o, --foutput FILE    Specify output file (default: default_output.txt)\n"
                                  << "  -t, --timeout NUM     Specify timeout value in seconds (default: 10)\n"
                                  << "  -d, --dtype TYPE      Specify char for MPI_CHAR, double MPI_DOUBLE or int for MPI_INT32 (default: double)\n"
                                  << "  -v, --verbose         Enable verbose mode\n";
                        // @formatter:on
                        return EXIT_SUCCESS;
                case 'm':
                        fmessages = optarg;
                        break;
                case 'o':
                        foutput = optarg;
                        break;
                case 'd':
                        dtype = optarg;
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
                if (dtype == "double") {
                        Scatterv<double> benchmark(fmessages);
                        benchmark.run(timeout, verbose);
                        benchmark.save_latencies(foutput, verbose);
                } else if (dtype == "int") {
                        Scatterv<int> benchmark(fmessages);
                        benchmark.run(timeout, verbose);
                        benchmark.save_latencies(foutput, verbose);
                } else if (dtype == "char") {
                        Scatterv<char> benchmark(fmessages);
                        benchmark.run(timeout, verbose);
                        benchmark.save_latencies(foutput, verbose);
                } else {
                        std::cerr << "Unknown dtype option: " << dtype << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        return EXIT_FAILURE;
                }
        } catch (const std::exception &e) {
                std::cerr << "ERROR: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                return EXIT_FAILURE;
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
}