import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_latencies(filename):
    data = pd.read_csv(filename)

    plt.figure(figsize=(10, 6))
    for rank, group in data.groupby("Rank"):
        plt.plot(group["Iteration"], group["Latency"], label=f"Rank {rank}")

    plt.xlabel("Iteration")
    plt.ylabel("Latency (seconds)")
    plt.title("Broadcast Latency by Rank")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    plot_latencies(args.filename)

