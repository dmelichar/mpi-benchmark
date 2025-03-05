import numpy as np
import argparse


def equal(nproc: int, val: int, m2m: bool = False, savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        size = nproc if not m2m else (nproc, nproc)
        values = np.full(size, val, dtype=np.uintp)
        f = f"{savedir}/{nproc}{'-m2m' if m2m else ''}-equal.csv"
        values = values.reshape(1, -1) if not m2m else values
        np.savetxt(f, values, delimiter=",", fmt="%d")
        return values, f


def normal(nproc: int, m2m: bool = False, savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        size = nproc if not m2m else (nproc, nproc)
        values = np.abs(np.random.normal(10, 15, size)).astype(np.uintp)
        f = f"{savedir}/{nproc}{'-m2m' if m2m else ''}-normal.csv"
        values = values.reshape(1, -1) if not m2m else values
        np.savetxt(f, values, delimiter=",", fmt="%d")
        return values, f


def exponential(nproc: int, m2m: bool = False, savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        size = nproc if not m2m else (nproc, nproc)
        values = np.random.exponential(50, size).astype(np.uintp)
        f = f"{savedir}/{nproc}{'-m2m' if m2m else ''}-exponential.csv"
        values = values.reshape(1, -1) if not m2m else values
        np.savetxt(f, values.reshape(1,-1), delimiter=",", fmt="%d")
        return values, f


def increasing(nproc: int, avg: int,  savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        values = np.floor((2 * avg * (np.arange(nproc) + 1)) / nproc).astype(np.uintp)
        f = f"{savedir}/{nproc}-increasing.csv"
        np.savetxt(f, values.reshape(1,-1), delimiter=",", fmt="%d")
        return values, f


def decreasing(nproc: int, avg: int,  savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        values = (np.floor((2 * avg * (nproc - np.arange(nproc))) / nproc) + 1).astype(np.uintp)
        f = f"{savedir}/{nproc}-decreasing.csv"
        np.savetxt(f, values.reshape(1,-1), delimiter=",", fmt="%d")
        return values, f


def zipfian(nproc: int, m2m: bool = False,  savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        size = nproc if not m2m else (nproc, nproc)
        values = np.random.zipf(2, size).astype(np.uintp)
        f = f"{savedir}/{nproc}{'-m2m' if m2m else ''}-zipfian.csv"
        values = values.reshape(1, -1) if not m2m else values
        np.savetxt(f, values.reshape(1,-1), delimiter=",", fmt="%d")
        return values, f


def uniform(nproc: int, avg: int, m2m: bool = False,  savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        size = nproc if not m2m else (nproc, nproc)
        values = np.random.randint(1, 2 * avg, size=size).astype(np.uintp)
        f = f"{savedir}/{nproc}{'-m2m' if m2m else ''}-uniform.csv"
        values = values.reshape(1, -1) if not m2m else values
        np.savetxt(f, values, delimiter=",", fmt="%d")
        return values, f


def bucket(nproc: int, avg: int, m2m: bool = False, savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        size = nproc if not m2m else (nproc, nproc)
        values = (avg // 2) + np.random.randint(1, avg, size=size).astype(np.uintp)
        f = f"{savedir}/{nproc}{'-m2m' if m2m else ''}-bucket.csv"
        values = values.reshape(1, -1) if not m2m else values
        np.savetxt(f, values.reshape(1,-1), delimiter=",", fmt="%d")
        return values, f


def spikes(nproc: int, avg: int, rho: float = 2, m2m: bool = False,  savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        size = nproc if not m2m else (nproc, nproc)
        values = np.random.choice([rho * avg, 1], size=size, p=[1 / rho, 1 - 1 / rho]).astype(np.uintp)
        f = f"{savedir}/{nproc}{'-m2m' if m2m else ''}-spikes.csv"
        values = values.reshape(1, -1) if not m2m else values
        np.savetxt(f, values, delimiter=",", fmt="%d")
        return values, f


def alternating(nproc: int, avg: int, m2m: bool = False, savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        values = np.array([avg + avg // 2 if i % 2 == 0 else avg - avg // 2 for i in range(nproc)]).astype(np.uintp)
        f = f"{savedir}/{nproc}{'-m2m' if m2m else ''}-alternating.csv"
        values = values.reshape(1, -1) if not m2m else np.tile(values, (nproc, 1))
        np.savetxt(f, values, delimiter=",", fmt="%d")
        return values, f


def two_blocks(nproc: int, avg: int, m2m: bool = False,  savedir: str = ".", seed: int = 42):
        np.random.seed(seed)
        size = nproc if not m2m else (nproc, nproc)
        values = np.zeros(size, dtype=np.uintp)
        values[0] = avg
        values[-1] = avg
        values = values.reshape(1,-1) if not m2m else values.reshape((nproc, nproc))
        f = f"{savedir}/{nproc}{'-m2m' if m2m else ''}-two-blocks.csv"
        np.savetxt(f, values, delimiter=",", fmt="%d")
        return values, f


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="CLI for generating block sizes and data distributions")

        subparsers = parser.add_subparsers(dest='command')

        # Subcommands for different distributions
        equal_parser = subparsers.add_parser('equal', help='Generate equal distribution')
        equal_parser.add_argument('nproc', type=int, help="Number of processors")
        equal_parser.add_argument('val', type=int, help="Value to fill the array")
        equal_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")
        equal_parser.add_argument('--seed', type=int, default=42, help="Random seed")
        equal_parser.add_argument("--savedir", type=str, default=".", help="Save file to dir")

        normal_parser = subparsers.add_parser('normal', help='Generate normal distribution')
        normal_parser.add_argument('nproc', type=int, help="Number of processors")
        normal_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")
        normal_parser.add_argument('--seed', type=int, default=42, help="Random seed")
        normal_parser.add_argument("--savedir", type=str, default=".", help="Save file to dir")

        exponential_parser = subparsers.add_parser('exponential', help='Generate exponential distribution')
        exponential_parser.add_argument('nproc', type=int, help="Number of processors")
        exponential_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")
        exponential_parser.add_argument('--seed', type=int, default=42, help="Random seed")
        exponential_parser.add_argument("--savedir", type=str, default=".", help="Save file to dir")

        increasing_parser = subparsers.add_parser('increasing', help='Generate increasing sequence')
        increasing_parser.add_argument('nproc', type=int, help="Number of processors")
        increasing_parser.add_argument('avg', type=int, help="Average block size")
        increasing_parser.add_argument('--seed', type=int, default=42, help="Random seed")
        increasing_parser.add_argument("--savedir", type=str, default=".", help="Save file to dir")

        decreasing_parser = subparsers.add_parser('decreasing', help='Decreasing block sizes')
        decreasing_parser.add_argument('nproc', type=int, help="Number of processors")
        decreasing_parser.add_argument('avg', type=int, help="Average block size")
        decreasing_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")

        zipfian_parser = subparsers.add_parser('zipfian', help='Generate zipfian distribution')
        zipfian_parser.add_argument('nproc', type=int, help="Number of processors")
        zipfian_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")
        zipfian_parser.add_argument('--seed', type=int, default=42, help="Random seed")
        zipfian_parser.add_argument("--savedir", type=str, default=".", help="Save file to dir")

        uniform_parser = subparsers.add_parser('uniform', help='Random block sizes')
        uniform_parser.add_argument('nproc', type=int, help="Number of processors")
        uniform_parser.add_argument('avg', type=int, help="Average block size")
        uniform_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")
        uniform_parser.add_argument("--savedir", type=str, default=".", help="Save file to dir")

        bucket_parser = subparsers.add_parser('bucket', help='Bucket block sizes')
        bucket_parser.add_argument('nproc', type=int, help="Number of processors")
        bucket_parser.add_argument('avg', type=int, help="Average block size")
        bucket_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")
        bucket_parser.add_argument("--savedir", type=str, default=".", help="Save file to dir")

        spikes_parser = subparsers.add_parser('spikes', help='Spikes block sizes')
        spikes_parser.add_argument('nproc', type=int, help="Number of processors")
        spikes_parser.add_argument('avg', type=int, help="Average block size")
        spikes_parser.add_argument('--rho', type=float, default=2, help="Spike factor")
        spikes_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")
        spikes_parser.add_argument("--savedir", type=str, default=".", help="Save file to dir")

        alternating_blocks_parser = subparsers.add_parser('alternating', help='Alternating block sizes')
        alternating_blocks_parser.add_argument('nproc', type=int, help="Number of processors")
        alternating_blocks_parser.add_argument('avg', type=int, help="Average block size")
        alternating_blocks_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")
        alternating_blocks_parser.add_argument("--savedir", type=str, default="", help="Save file to dir")

        two_blocks_parser = subparsers.add_parser('two-blocks', help='Two-blocks block sizes')
        two_blocks_parser.add_argument('nproc', type=int, help="Number of processors")
        two_blocks_parser.add_argument('avg', type=int, help="Average block size")
        two_blocks_parser.add_argument('--m2m', action='store_true', help="Many-to-many distribution")
        two_blocks_parser.add_argument("--savedir", type=str, default="", help="Save file to dir")

        args = parser.parse_args()

        if args.command == 'equal':
                equal(args.nproc, args.val, args.m2m, args.savedir, args.seed)
        elif args.command == 'normal':
                normal(args.nproc, args.m2m, args.savedir,  args.seed)
        elif args.command == 'exponential':
                exponential(args.nproc, args.m2m, args.savedir, args.seed)
        elif args.command == 'increasing':
                increasing(args.nproc, args.avg, args.savedir, args.seed)
        elif args.command == 'decreasing':
                decreasing(args.nproc, args.avg, args.savedir, args.seed)
        elif args.command == 'zipfian':
                zipfian(args.nproc, args.m2m, args.savedir,  args.seed)
        elif args.command == 'uniform':
                uniform(args.nproc, args.avg, args.m2m, args.savedir,  args.seed)
        elif args.command == 'bucket':
                bucket(args.nproc, args.avg, args.m2m, args.savedir,  args.seed)
        elif args.command == 'spikes':
                spikes(args.nproc, args.avg, args.rho, args.m2m, args.savedir,  args.seed)
        elif args.command == 'alternating':
                alternating(args.nproc, args.avg, args.m2m, args.savedir,  args.seed)
        elif args.command == "two_blocks":
                two_blocks(args.nproc, args.avg, args.m2m, args.savedir,  args.seed)
