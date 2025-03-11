#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <deque>
#include <numeric>
#include <cstdlib>
#include <unistd.h>

#include <mpi.h>

#define MSG_COUNT 262144
#define MAX_SECONDS 1
#define ALIGNMENT 8

void osu_timer(const int rank, const int csize, const int size = 2640)
{
        double *sbuffer = nullptr;
        double *rbuffer = nullptr;
        int *sendcounts = nullptr;
        int *displs = nullptr;

        size_t alignment = sysconf(_SC_PAGESIZE);

        if (rank == 0) {
                if (posix_memalign(reinterpret_cast<void **>(&sbuffer),
                                   alignment,
                                   MSG_COUNT * csize * sizeof(double)) != 0) {
                        std::cerr << "Failed to allocate memory for sbuffer!" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                for (int i = 0; i < MSG_COUNT * csize; ++i) {
                        sbuffer[i] = i; // Fill the send buffer with data
                }
                sendcounts = new int[4]{MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
                displs = new int[4]{0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};
        }

        if (posix_memalign(reinterpret_cast<void **>(&rbuffer), alignment, MSG_COUNT * sizeof(double)) != 0) {
                std::cerr << "Failed to allocate memory for rbuffer!" << std::endl;
                free(sbuffer); // Clean up if allocation fails
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double timer = 0.0;
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displs,
                             MPI_DOUBLE,
                             rbuffer,
                             MSG_COUNT,
                             MPI_DOUBLE,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                timer += t_stop - t_start;
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency = (timer * 1e6) / 2640;
        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }

        delete[] sbuffer;
        delete[] rbuffer;
        if (rank == 0) {
                delete[] sendcounts;
                delete[] displs;
        }

}


void osu_dq(const int rank, const int csize, const int size = 2640)
{
        double *sbuffer = nullptr;
        double *rbuffer = nullptr;
        int *sendcounts = nullptr;
        int *displs = nullptr;

        size_t alignment = sysconf(_SC_PAGESIZE);

        if (rank == 0) {
                if (posix_memalign(reinterpret_cast<void **>(&sbuffer),
                                   alignment,
                                   MSG_COUNT * csize * sizeof(double)) != 0) {
                        std::cerr << "Failed to allocate memory for sbuffer!" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                for (int i = 0; i < MSG_COUNT * csize; ++i) {
                        sbuffer[i] = i; // Fill the send buffer with data
                }
                sendcounts = new int[4]{MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
                displs = new int[4]{0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};
        }

        if (posix_memalign(reinterpret_cast<void **>(&rbuffer), alignment, MSG_COUNT * sizeof(double)) != 0) {
                std::cerr << "Failed to allocate memory for rbuffer!" << std::endl;
                free(sbuffer); // Clean up if allocation fails
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        std::deque<double> starts;
        std::deque<double> ends;

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displs,
                             MPI_DOUBLE,
                             rbuffer,
                             MSG_COUNT,
                             MPI_DOUBLE,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                starts.push_back(t_start);
                ends.push_back(t_stop);
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency[size];
        for (int i = 0; i < size; ++i) {
                latency[i] = (ends[i] - starts[i]) * 1e6;
        }
        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }

}


void osu_vector(const int rank, const int csize, const int size = 2640)
{
        double *sbuffer = nullptr;
        double *rbuffer = nullptr;
        int *sendcounts = nullptr;
        int *displs = nullptr;

        size_t alignment = sysconf(_SC_PAGESIZE);

        if (rank == 0) {
                if (posix_memalign(reinterpret_cast<void **>(&sbuffer),
                                   alignment,
                                   MSG_COUNT * csize * sizeof(double)) != 0) {
                        std::cerr << "Failed to allocate memory for sbuffer!" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                for (int i = 0; i < MSG_COUNT * csize; ++i) {
                        sbuffer[i] = i; // Fill the send buffer with data
                }
                sendcounts = new int[4]{MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
                displs = new int[4]{0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};
        }

        if (posix_memalign(reinterpret_cast<void **>(&rbuffer), alignment, MSG_COUNT * sizeof(double)) != 0) {
                std::cerr << "Failed to allocate memory for rbuffer!" << std::endl;
                free(sbuffer); // Clean up if allocation fails
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        std::vector<double> starts;
        std::vector<double> ends;

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displs,
                             MPI_DOUBLE,
                             rbuffer,
                             MSG_COUNT,
                             MPI_DOUBLE,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                starts.push_back(t_start);
                ends.push_back(t_stop);
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency[size];
        for (int i = 0; i < size; ++i) {
                latency[i] = (ends[i] - starts[i]) * 1e6;
        }
        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }

}

void stack_timer(const int rank, const int csize, const int size = 2640)
{
        double sbuffer[MSG_COUNT * csize];
        double rbuffer[MSG_COUNT];

        int sendcounts[4] = {MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
        int displacements[4] = {0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};

        MPI_Barrier(MPI_COMM_WORLD);
        double timer = 0.0;
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displacements,
                             MPI_DOUBLE,
                             rbuffer,
                             sendcounts[rank],
                             MPI_DOUBLE,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                timer += t_stop - t_start;
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency = (timer * 1e6) / 2640;

        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }
}


void stack_dq(const int rank, const int csize, const int size = 2640)
{
        std::deque<double> starts;
        std::deque<double> ends;

        double sbuffer[MSG_COUNT * csize];
        double rbuffer[MSG_COUNT];

        int sendcounts[4] = {MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
        int displacements[4] = {0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};

        MPI_Barrier(MPI_COMM_WORLD);
        double timer = 0.0;
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displacements,
                             MPI_DOUBLE,
                             rbuffer,
                             sendcounts[rank],
                             MPI_DOUBLE,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                starts.push_back(t_start);
                ends.push_back(t_stop);
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency[size];
        for (int i = 0; i < size; ++i) {
                latency[i] = (ends[i] - starts[i]) * 1e6;
        }
        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }
}


void stack_vector(const int rank, const int csize, const int size = 2640)
{
        std::vector<double> starts;
        std::vector<double> ends;

        double sbuffer[MSG_COUNT * csize];
        double rbuffer[MSG_COUNT];

        int sendcounts[4] = {MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
        int displacements[4] = {0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};

        MPI_Barrier(MPI_COMM_WORLD);
        double timer = 0.0;
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displacements,
                             MPI_DOUBLE,
                             rbuffer,
                             sendcounts[rank],
                             MPI_DOUBLE,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                starts.push_back(t_start);
                ends.push_back(t_stop);
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency[size];
        for (int i = 0; i < size; ++i) {
                latency[i] = (ends[i] - starts[i]) * 1e6;
        }
        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }
}


void heap_timer(const int rank, const int csize, const int size = 2640)
{
        auto *sbuffer = new double[MSG_COUNT * csize];
        auto *rbuffer = new double[MSG_COUNT];

        int sendcounts[4] = {MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
        int displacements[4] = {0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};

        MPI_Barrier(MPI_COMM_WORLD);
        double timer = 0.0;
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displacements,
                             MPI_DOUBLE,
                             rbuffer,
                             sendcounts[rank],
                             MPI_DOUBLE,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                timer += t_stop - t_start;
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency = (timer * 1e6) / 2640;

        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }

        delete[] sbuffer;
        delete[] rbuffer;
}

void heap_dq(const int rank, const int csize, const int size = 2640)
{
        auto *sbuffer = new double[MSG_COUNT * csize];
        auto *rbuffer = new double[MSG_COUNT];

        std::deque<double> starts;
        std::deque<double> ends;

        int sendcounts[4] = {MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
        int displacements[4] = {0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};

        MPI_Barrier(MPI_COMM_WORLD);
        double timer = 0.0;
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displacements,
                             MPI_DOUBLE,
                             rbuffer,
                             sendcounts[rank],
                             MPI_DOUBLE,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                starts.push_back(t_start);
                ends.push_back(t_stop);
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency[size];
        for (int i = 0; i < size; ++i) {
                latency[i] = (ends[i] - starts[i]) * 1e6;
        }
        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }

        delete[] sbuffer;
        delete[] rbuffer;
}


void heap_vector(const int rank, const int csize, const int size = 2640)
{
        auto *sbuffer = new double[MSG_COUNT * csize];
        auto *rbuffer = new double[MSG_COUNT];

        std::vector<double> starts;
        std::vector<double> ends;

        int sendcounts[4] = {MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
        int displacements[4] = {0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};

        MPI_Barrier(MPI_COMM_WORLD);
        double timer = 0.0;
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displacements,
                             MPI_DOUBLE,
                             rbuffer,
                             sendcounts[rank],
                             MPI_DOUBLE,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                starts.push_back(t_start);
                ends.push_back(t_stop);
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency[size];
        for (int i = 0; i < size; ++i) {
                latency[i] = (ends[i] - starts[i]) * 1e6;
        }
        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }

        delete[] sbuffer;
        delete[] rbuffer;
}


class VectorTimer {
        std::vector<double> sbuffer;
        std::vector<double> rbuffer;
        std::vector<int> displs;
        std::vector<int> sendcounts;
        int rank;
        int csize;
        int size;

public:
        VectorTimer(const int rank, const int csize, const int size = 2640)
        {
                this->rank = rank;
                this->csize = csize;
                this->size = size;

                this->sbuffer.resize(MSG_COUNT * csize);
                this->rbuffer.resize(MSG_COUNT);
                std::ranges::fill(sbuffer, 0.0);
                std::ranges::fill(rbuffer, 0.0);

                this->displs.resize(csize);
                this->sendcounts.resize(csize);

                this->sendcounts = {MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
                this->displs = {0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};
        }

        void run()
        {

                MPI_Barrier(MPI_COMM_WORLD);
                double timer = 0.0;
                for (int i = 0; i < size; i++) {
                        const double t_start = MPI_Wtime();
                        MPI_Scatterv(this->sbuffer.data(),
                                     this->sendcounts.data(),
                                     this->displs.data(),
                                     MPI_DOUBLE,
                                     this->rbuffer.data(),
                                     this->sendcounts[this->rank],
                                     MPI_DOUBLE,
                                     0,
                                     MPI_COMM_WORLD);
                        const double t_stop = MPI_Wtime();

                        timer += t_stop - t_start;
                        MPI_Barrier(MPI_COMM_WORLD);
                }
                MPI_Barrier(MPI_COMM_WORLD);

                double latency = (timer * 1e6) / 2640;
                double min_time, max_time, avg_time;
                MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
                MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                avg_time = avg_time / csize;

                if (rank == 0) {
                        std::cout << "Minimum latency: " << min_time << std::endl;
                        std::cout << "Maximum latency: " << max_time << std::endl;
                        std::cout << "Average latency: " << avg_time << std::endl;
                        std::cout << "Iterations: " << this->size << std::endl;
                }

        }
};


class VectorDQ {
        std::vector<double> sbuffer;
        std::vector<double> rbuffer;
        std::vector<int> displs;
        std::vector<int> sendcounts;
        int rank;
        int csize;
        int size;

public:
        VectorDQ(const int rank, const int csize, const int size = 2640)
        {
                this->rank = rank;
                this->csize = csize;
                this->size = size;

                if (rank == 0) {
                        this->sbuffer.resize(MSG_COUNT * csize);
                } else {
                        this->sbuffer.resize(1);
                }
                this->rbuffer.resize(MSG_COUNT);
                this->displs.resize(csize);
                this->sendcounts.resize(csize);

                this->sendcounts = {MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
                this->displs = {0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};
        }

        void run()
        {

                std::deque<double> starts;
                std::deque<double> ends;

                MPI_Barrier(MPI_COMM_WORLD);
                double timer = 0.0;
                for (int i = 0; i < 2640; i++) {
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
                }
                MPI_Barrier(MPI_COMM_WORLD);

                double latency[size];
                for (int i = 0; i < size; ++i) {
                        latency[i] = (ends[i] - starts[i]) * 1e6;
                }
                double min_time, max_time, avg_time;
                MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
                MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                avg_time = avg_time / csize;

                if (rank == 0) {
                        std::cout << "Minimum latency: " << min_time << std::endl;
                        std::cout << "Maximum latency: " << max_time << std::endl;
                        std::cout << "Average latency: " << avg_time << std::endl;
                        std::cout << "Iterations: " << size << std::endl;
                }
        }
};

class VectorVector {
        std::vector<double> sbuffer;
        std::vector<double> rbuffer;
        std::vector<int> displs;
        std::vector<int> sendcounts;
        int rank;
        int csize;
        int size;

public:
        VectorVector(const int rank, const int csize, const int size = 2640)
        {
                this->rank = rank;
                this->csize = csize;
                this->size = size;

                if (rank == 0) {
                        this->sbuffer.resize(MSG_COUNT * csize);
                } else {
                        this->sbuffer.resize(1);
                }
                this->rbuffer.resize(MSG_COUNT);
                this->displs.resize(csize);
                this->sendcounts.resize(csize);

                this->sendcounts = {MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
                this->displs = {0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};
        }

        void run()
        {

                std::vector<double> starts;
                std::vector<double> ends;

                MPI_Barrier(MPI_COMM_WORLD);
                double timer = 0.0;
                for (int i = 0; i < 2640; i++) {
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
                }
                MPI_Barrier(MPI_COMM_WORLD);

                double latency[size];
                for (int i = 0; i < size; ++i) {
                        latency[i] = (ends[i] - starts[i]) * 1e6;
                }
                double min_time, max_time, avg_time;
                MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
                MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                avg_time = avg_time / csize;

                if (rank == 0) {
                        std::cout << "Minimum latency: " << min_time << std::endl;
                        std::cout << "Maximum latency: " << max_time << std::endl;
                        std::cout << "Average latency: " << avg_time << std::endl;
                        std::cout << "Iterations: " << size << std::endl;
                }
        }
};

void char_timer(const int rank, const int csize, const int size = 2640)
{
        char *sbuffer = nullptr;
        char *rbuffer = nullptr;
        int *sendcounts = nullptr;
        int *displs = nullptr;

        size_t alignment = sysconf(_SC_PAGESIZE);

        if (rank == 0) {
                if (posix_memalign(reinterpret_cast<void **>(&sbuffer),
                                   alignment,
                                   MSG_COUNT * csize * sizeof(char)) != 0) {
                        std::cerr << "Failed to allocate memory for sbuffer!" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                for (int i = 0; i < MSG_COUNT * csize; ++i) {
                        sbuffer[i] = 'a'; // Fill the send buffer with dummy data (e.g., 'a')
                }
                sendcounts = new int[4]{MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
                displs = new int[4]{0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};
        }

        if (posix_memalign(reinterpret_cast<void **>(&rbuffer), alignment, MSG_COUNT * sizeof(char)) != 0) {
                std::cerr << "Failed to allocate memory for rbuffer!" << std::endl;
                free(sbuffer); // Clean up if allocation fails
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double timer = 0.0;
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displs,
                             MPI_CHAR,
                             rbuffer,
                             MSG_COUNT,
                             MPI_CHAR,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                timer += t_stop - t_start;
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency = (timer * 1e6) / 2640;
        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }

        delete[] sbuffer;
        delete[] rbuffer;
        if (rank == 0) {
                delete[] sendcounts;
                delete[] displs;
        }
}

void char_dq(const int rank, const int csize, const int size = 2640)
{
        std::deque<double> starts;
        std::deque<double> ends;

        char *sbuffer = nullptr;
        char *rbuffer = nullptr;
        int *sendcounts = nullptr;
        int *displs = nullptr;

        size_t alignment = sysconf(_SC_PAGESIZE);

        if (rank == 0) {
                if (posix_memalign(reinterpret_cast<void **>(&sbuffer),
                                   alignment,
                                   MSG_COUNT * csize * sizeof(char)) != 0) {
                        std::cerr << "Failed to allocate memory for sbuffer!" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                for (int i = 0; i < MSG_COUNT * csize; ++i) {
                        sbuffer[i] = 'a'; // Fill the send buffer with dummy data (e.g., 'a')
                }
                sendcounts = new int[4]{MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
                displs = new int[4]{0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};
        }

        if (posix_memalign(reinterpret_cast<void **>(&rbuffer), alignment, MSG_COUNT * sizeof(char)) != 0) {
                std::cerr << "Failed to allocate memory for rbuffer!" << std::endl;
                free(sbuffer); // Clean up if allocation fails
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displs,
                             MPI_CHAR,
                             rbuffer,
                             MSG_COUNT,
                             MPI_CHAR,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                starts.push_back(t_start);
                ends.push_back(t_stop);
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency[size];
        for (int i = 0; i < size; ++i) {
                latency[i] = (ends[i] - starts[i]) * 1e6;
        }

        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }

        delete[] sbuffer;
        delete[] rbuffer;
        if (rank == 0) {
                delete[] sendcounts;
                delete[] displs;
        }
}


void char_vector(const int rank, const int csize, const int size = 2640)
{
        std::vector<double> starts;
        std::vector<double> ends;

        char *sbuffer = nullptr;
        char *rbuffer = nullptr;
        int *sendcounts = nullptr;
        int *displs = nullptr;

        size_t alignment = sysconf(_SC_PAGESIZE);

        if (rank == 0) {
                if (posix_memalign(reinterpret_cast<void **>(&sbuffer),
                                   alignment,
                                   MSG_COUNT * csize * sizeof(char)) != 0) {
                        std::cerr << "Failed to allocate memory for sbuffer!" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                for (int i = 0; i < MSG_COUNT * csize; ++i) {
                        sbuffer[i] = 'a'; // Fill the send buffer with dummy data (e.g., 'a')
                }
                sendcounts = new int[4]{MSG_COUNT, MSG_COUNT, MSG_COUNT, MSG_COUNT};
                displs = new int[4]{0, MSG_COUNT, 2 * MSG_COUNT, 3 * MSG_COUNT};
        }

        if (posix_memalign(reinterpret_cast<void **>(&rbuffer), alignment, MSG_COUNT * sizeof(char)) != 0) {
                std::cerr << "Failed to allocate memory for rbuffer!" << std::endl;
                free(sbuffer); // Clean up if allocation fails
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < 2640; i++) {
                const double t_start = MPI_Wtime();
                MPI_Scatterv(sbuffer,
                             sendcounts,
                             displs,
                             MPI_CHAR,
                             rbuffer,
                             MSG_COUNT,
                             MPI_CHAR,
                             0,
                             MPI_COMM_WORLD);
                const double t_stop = MPI_Wtime();

                starts.push_back(t_start);
                ends.push_back(t_stop);
                MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double latency[size];
        for (int i = 0; i < size; ++i) {
                latency[i] = (ends[i] - starts[i]) * 1e6;
        }

        double min_time, max_time, avg_time;
        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_time = avg_time / csize;

        if (rank == 0) {
                std::cout << "Minimum latency: " << min_time << std::endl;
                std::cout << "Maximum latency: " << max_time << std::endl;
                std::cout << "Average latency: " << avg_time << std::endl;
                std::cout << "Iterations: " << size << std::endl;
        }

        delete[] sbuffer;
        delete[] rbuffer;
        if (rank == 0) {
                delete[] sendcounts;
                delete[] displs;
        }
}


int main(int argc, char *argv[])
{
        int rank;
        int csize;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &csize);

        int opt;
        std::string mode = "timer";  // Default mode is timer.

        // Parse arguments
        while ((opt = getopt(argc, argv, "m:")) != -1) {
                switch (opt) {
                case 'm':  // -m <mode> argument
                        mode = optarg;
                        break;
                default:
                        std::cerr << "Usage: " << argv[0] << " -m <mode>" << std::endl;
                        exit(EXIT_FAILURE);
                }
        }

        if (mode == "timer") {
                if (rank == 0) {
                        std::cout << "===> OSU Timer" << std::endl;
                }
                osu_timer(rank, csize);
                if (rank == 0) {
                        std::cout << "===> Char Timer" << std::endl;
                }
                char_timer(rank, csize);
                if (rank == 0) {
                        std::cout << "===> Stack timer" << std::endl;
                }
                stack_timer(rank, csize);
                if (rank == 0) {
                        std::cout << "===> Heap timer" << std::endl;
                }
                heap_timer(rank, csize);
                if (rank == 0) {
                        std::cout << "==> Vector timer" << std::endl;
                }
                VectorTimer c(rank, csize, 2640);
                c.run();
        } else if (mode == "dq") {
                if (rank == 0) {
                        std::cout << std::endl;
                        std::cout << "===> OSU DQ" << std::endl;
                }
                osu_dq(rank, csize);
                if (rank == 0) {
                        std::cout << "===> Char DQ" << std::endl;
                }
                char_dq(rank, csize);
                if (rank == 0) {
                        std::cout << "==> Stack DQ" << std::endl;
                }
                stack_dq(rank, csize);
                if (rank == 0) {
                        std::cout << "==> Heap DQ" << std::endl;
                }
                heap_dq(rank, csize);
                if (rank == 0) {
                        std::cout << "==> Vector DQ" << std::endl;
                }
                VectorDQ d(rank, csize, 2640);
                d.run();
        } else if (mode == "vector") {
                if (rank == 0) {
                        std::cout << std::endl;
                        std::cout << "===> OSU Vector" << std::endl;
                }
                osu_vector(rank, csize);
                if (rank == 0) {
                        std::cout << "===> Char Vector" << std::endl;
                }
                char_vector(rank, csize);
                if (rank == 0) {
                        std::cout << "==> Stack Vector" << std::endl;
                }
                stack_vector(rank, csize);
                if (rank == 0) {
                        std::cout << "==> Heap Vector" << std::endl;
                }
                heap_vector(rank, csize);
                if (rank == 0) {
                        std::cout << "==> Vector Vector" << std::endl;
                }
                VectorVector t(rank, csize, 2640);
                t.run();
        }

        /* TODO Setup data structures and do memory checks of stack/heap */

        /* TODO Synchronized time-based loop with Scatterv calls */

        /* TODO Get start/end times of all processes and do basic statistics */

        /* TODO Delete data structures */

        /* TODO Write to file */

        MPI_Finalize();

        return EXIT_SUCCESS;
}