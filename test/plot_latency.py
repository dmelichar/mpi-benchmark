import matplotlib.pyplot as plt
import argparse
import csv

from collections import defaultdict


def plot_latencies(filename):
        iterations, ranks, starts, ends = [], [], [], []

        with open(filename, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                        iterations.append(int(row["Iteration"]))
                        starts.append(float(row["Starttime"]))
                        ends.append(float(row["Endtime"]))
                        ranks.append(int(row["Rank"]))

        latencies = [(end-start)*1e6 for start, end in zip(starts,ends)]

        # Create a figure with two subplots (line plot and boxplot)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ######## Lineplot
        # Line plot showing max latency over all processes at each iteration

        sorted_iterations = sorted(set(iterations))
        niterations = sorted_iterations[-1] + 1

        ax1.scatter(sorted_iterations, latencies[0:niterations], label=f"Rank 0", color='r', s=10, marker="o")
        ax1.scatter(sorted_iterations, latencies[niterations:2*niterations], label=f"Rank 1", color='b', s=10, marker="o")
        ax1.scatter(sorted_iterations, latencies[2*niterations:3*niterations], label=f"Rank 2", color='g', s=10, marker="o")
        ax1.scatter(sorted_iterations, latencies[3*niterations:], label=f"Rank 3", color='y', s=10, marker="o")

        # Plot details
        ax1.set_title("Latency by Rank Over Iterations")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Max Latency (seconds)")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.6)


        #### Boxplot

        data = defaultdict(list)
        for rank, latency in zip(ranks, latencies):
                data[rank].append(latency)

        ax2.set_title("Boxplot: Latency Distribution by Rank")
        ax2.set_xlabel("Rank")
        ax2.set_ylabel("Latency (seconds)")

        ax2.boxplot([data[rank] for rank in sorted(data.keys())],
                    tick_labels=[f"Rank {rank}" for rank in sorted(data.keys())])
        ax2.grid(True, linestyle="--", alpha=0.6)

        for tick in ax2.get_xticklabels():
                tick.set_rotation(45)


        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("filename")
        plot_latencies(parser.parse_args().filename)
