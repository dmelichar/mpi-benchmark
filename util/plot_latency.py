import pandas as pd
import matplotlib.pyplot as plt

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
    csv_filename = "latencies.csv"  # Adjust this if the file has a different name
    plot_latencies(csv_filename)

