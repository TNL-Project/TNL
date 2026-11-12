#!/usr/bin/env python3
"""Reference implementations of the ALS benchmark.

Runs alternating least squares for implicit feedback of the implicit library with the
same initial factors as tnl-benchmark-als (see getInitialFactors in ALSBenchmark.h) and
appends the results to the log file in the format of the TNL benchmarks:

- implicit-cpu: implicit.cpu.als.AlternatingLeastSquares with the conjugate gradient
  solver (Cython, OpenMP),
- implicit-cuda: implicit.gpu.als.AlternatingLeastSquares (CUDA kernels and cuBLAS).

Only the iterations are measured (by the callback of fit), not the conversion of the
matrix and its transfer to the GPU. The loss of the final factors is computed by the
CPU implementation in both cases.
"""

import argparse
import json

import numpy as np
import scipy.io
import scipy.sparse


def initial_factors(rows, factors, seed):
    """The same as getInitialFactors in ALSBenchmark.h."""
    with np.errstate(over="ignore"):
        z = np.arange(rows * factors, dtype=np.uint64) * np.uint64(
            0x9E3779B97F4A7C15
        ) + np.uint64(seed)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    h = (z >> np.uint64(40)).astype(np.float64) / float(1 << 24)
    return ((h - 0.5) / factors).astype(np.float32).reshape(rows, factors)


def run_implicit(R, factors, iterations, regularization, cg_steps, device):
    import implicit.cpu._als
    import implicit.gpu

    X0 = initial_factors(R.shape[0], factors, 1)
    Y0 = initial_factors(R.shape[1], factors, 2)
    if device == "cuda":
        import implicit.gpu.als

        model = implicit.gpu.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=0,
        )
        model.user_factors = implicit.gpu.Matrix(X0)
        model.item_factors = implicit.gpu.Matrix(Y0)
    else:
        import implicit.cpu.als

        model = implicit.cpu.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_cg=True,
            random_state=0,
        )
        model.user_factors = X0.copy()
        model.item_factors = Y0.copy()
    model.cg_steps = cg_steps
    times = []
    model.fit(
        R,
        show_progress=False,
        callback=lambda iteration, elapsed, loss: times.append(elapsed),
    )
    if device == "cuda":
        model = model.to_cpu()
    loss = implicit.cpu._als.calculate_loss(
        R, model.user_factors, model.item_factors, regularization
    )
    return sum(times), loss


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--factors", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--cg-steps", type=int, default=3)
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["implicit-cpu", "implicit-cuda"],
        choices=["implicit-cpu", "implicit-cuda"],
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="the minimal time of this number of runs is logged",
    )
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    R = scipy.sparse.csr_matrix(scipy.io.mmread(args.input_file), dtype=np.float32)
    R.sort_indices()

    with open(args.log_file, "a") as log:
        for factors in args.factors:
            for implementation in args.implementations:
                device = implementation.split("-")[1]
                runs = [
                    run_implicit(
                        R,
                        factors,
                        args.iterations,
                        args.regularization,
                        args.cg_steps,
                        device,
                    )
                    for _ in range(args.loops)
                ]
                elapsed, loss = min(runs)
                record = {
                    "operation": "ALS",
                    "matrix": args.input_file,
                    "users": str(R.shape[0]),
                    "items": str(R.shape[1]),
                    "nonzeros": str(R.nnz),
                    "precision": "float",
                    "factors": str(factors),
                    "iterations": str(args.iterations),
                    "regularization": f"{args.regularization:f}",
                    "cg steps": str(args.cg_steps),
                    "solver": "",
                    "device": "cuda" if device == "cuda" else "host",
                    "performer": implementation,
                    "time": f"{elapsed:e}",
                    "loops": str(args.loops),
                    "time per iteration": f"{elapsed / args.iterations:e}",
                    "loss": f"{loss:e}",
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(
                    implementation,
                    factors,
                    f"{elapsed / args.iterations * 1e3:.3f} ms/iteration",
                    f"loss {loss:.6e}",
                )


if __name__ == "__main__":
    main()
