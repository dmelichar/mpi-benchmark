import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np

from collections import defaultdict

def plot_latencies(filename):
    iterations, latencies, ranks = [], [], []

    # Reading data from CSV
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            iterations.append(int(row["Iteration"]))
            latencies.append(float(row["Latency"]))
            ranks.append(int(row["Rank"]))

    # Prepare the data in a format suitable for both plots
    data = defaultdict(list)
    for rank, latency in zip(ranks, latencies):
        data[rank].append(latency)

    # Create a figure with two subplots (line plot and boxplot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Line plot with confidence intervals
    ax1.set_title("Line Plot: Latency by Iteration and Rank")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Latency (seconds)")

    unique_ranks = sorted(set(ranks))
    for rank in unique_ranks:
        rank_iterations = [it for it, r in zip(iterations, ranks) if r == rank]
        rank_latencies = [lat for lat, r in zip(latencies, ranks) if r == rank]

        # Calculate mean and standard error of the mean (SEM)
        mean_latency = np.mean(rank_latencies)
        sem_latency = np.std(rank_latencies) / np.sqrt(len(rank_latencies))

        # Plot the line with shaded confidence intervals (using standard error)
        ax1.plot(rank_iterations, [mean_latency] * len(rank_iterations), label=f"Rank {rank}")
        ax1.fill_between(rank_iterations,
                         [mean_latency - sem_latency] * len(rank_iterations),
                         [mean_latency + sem_latency] * len(rank_iterations),
                         alpha=0.2)

    ax1.set_yscale("log")
    ax1.legend(loc="right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Boxplot
    ax2.set_title("Boxplot: Latency Distribution by Rank")
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Latency (seconds)")

    # Boxplot for latency by rank
    ax2.boxplot([data[rank] for rank in sorted(data.keys())], tick_labels=[f"Rank {rank}" for rank in sorted(data.keys())])

    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Rotate x-axis labels for boxplot if many labels are present
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)

    # Adjust layout to prevent label overlap
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    plot_latencies(parser.parse_args().filename)
