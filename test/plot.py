import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import csv
import pathlib
import seaborn as sns
import tqdm

from collections import defaultdict

sns.set_style("whitegrid")

def plot_dir(dirname: str):
    base = pathlib.Path(dirname)
    files = [f for f in list(base.glob("*")) if not f.name.split("-")[0].isdigit()]
    #files = [f for f in files if "1" in f.name.split("-")[-1]]
    files = sorted(files)
    
    fig, axes = plt.subplots(len(files), 2, figsize=(18, 6*len(files)))
    for f, ax in zip(files, axes):
        main(f, ax)
    fig.tight_layout()
    fig.savefig("scatterv.png", dpi=300)    
    

def main(filename, ax):
    q = 0.95
    head = {"Rank": np.int16, 
            "Iteration": np.int32,
            "Starttime": np.float64,
            "Endtime": np.float64}
    
    n, _ = filename.name.split(".")
    df = pd.read_csv(filename, dtype=head)
    df["Latency"] = (df["Endtime"] - df["Starttime"]) * 1e6
    
    q95 = df[(df["Latency"] <= df["Latency"].quantile(q)) & (df["Latency"] >= df["Latency"].quantile(float(format(1-q, ".2f"))))].copy()
    q95["Avg Latency"] = q95["Latency"].rolling(window=1000).mean()
    
    sns.violinplot(
        data=q95,
        x="Rank", 
        y="Latency", 
        ax=axes[0]
    )
    ax[0].set_title(f"{n}: Violin plot")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Latency")
    
    sns.lineplot(
        data=q95,
        x="Iteration",
        y="Avg Latency",
        hue="Rank",
        lw=1,
        err_style="band",
        errorbar="sd",
        ax=axes[1]
    )
    ax[1].set_title(f"{n}: Average Latency per iteration")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Average Latency")


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("filename")
        args = parser.parse_args()
        fig, ax = plt.subplots(1, 2, figsize=(18, 6)))
        main(args.filename, ax)
        fig.show()
