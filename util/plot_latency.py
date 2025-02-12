import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np

from collections import defaultdict


def plot_latencies(filename):
        iterations, latencies, ranks = [], [], []

        with open(filename, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                        iterations.append(int(row["Iteration"]))
                        latencies.append(float(row["Latency"]))
                        ranks.append(int(row["Rank"]))

        # Create a figure with two subplots (line plot and boxplot)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ######## Lineplot
        # Line plot showing max latency over all processes at each iteration

        iter_to_latencies = defaultdict(list)
        for it, latency in zip(iterations, latencies):
                iter_to_latencies[it].append(latency)

        sorted_iterations = sorted(iter_to_latencies.keys())
        max_latencies = [max(iter_to_latencies[it]) for it in sorted_iterations]

        # Plot the maximum latency for each iteration as a continuous line
        ax1.plot(sorted_iterations, max_latencies, label="Max Latency")

        ax1.set_title("Max Latency Over All Processes by Iteration")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Max Latency (seconds)")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.6)

        ##### Boxplot

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
