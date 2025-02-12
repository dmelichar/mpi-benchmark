import numpy as np
import argparse


def dgp(nproc, m2m, dist, filename, seed):
        np.random.seed(seed)

        if not m2m:
                # one to many, e.g. scatter, allgather
                # if dist is equal then scatter otherwise scatter
                size = nproc
        else:
                # many to many, e.g. alltoall
                size = (nproc, nproc)

        if dist == "normal":
                values = np.abs(np.random.normal(10, 15, size)).astype(np.uintp)
        elif dist == "exponential":
                values = np.random.exponential(50, size).astype(np.uintp)
        elif dist == "increasing":
                if not m2m:
                        values = np.arange(size, dtype=np.uintp)
                else:
                        values = np.arange(np.prod(size), dtype=np.uintp).reshape(size)
        elif dist == "zipfian":
                values = np.random.zipf(2, size).astype(np.uintp)
        elif dist == "equal":
                values = np.full(size, 10, dtype=np.uintp)
        else:
                raise ValueError("Unsupported dist.")

        #  Persist data to disk if filename is provided
        f = filename if filename else f"{nproc}{'-m2m' if m2m else ''}-{dist}.csv"
        values = values.reshape(1, -1) if not m2m else values
        np.savetxt(f, values, delimiter=",", fmt="%d")


if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument("--nproc", type=int, required=True, help="Number of processes")
        parser.add_argument(
                "--m2m",
                action="store_true",
                help="Enable many-to-many communication. This creates a matrix m_ij where the process i sends m_ij messages to the process j. (default is False)",
        )
        parser.add_argument(
                "--dist",
                type=str,
                choices=["equal", "normal", "exponential", "increasing", "zipfian"],
                default="equal",
                help="Distribution type (default is 'equal')",
        )
        parser.add_argument(
                "--filename",
                type=str,
                default=None,
                help="File name (default format 'nproc-m2m-dist.csv')",
        )
        parser.add_argument("--seed", type=int, default=42, help="Random seed (default is 42)")

        args = parser.parse_args()

        dgp(nproc=args.nproc,
            m2m=args.m2m,
            dist=args.dist,
            filename=args.filename,
            seed=args.seed,
            )
