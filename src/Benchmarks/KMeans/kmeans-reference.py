#!/usr/bin/env python3
"""Reference implementations of the k-means benchmark.

Runs Lloyd's algorithm with the same initialization as tnl-benchmark-kmeans (the points
with the indexes i * n / K) and a fixed number of iterations, and appends the results to
the log file in the format of the TNL benchmarks:

- sklearn: sklearn.cluster.KMeans(algorithm="lloyd") on the CSR matrix (CPU, OpenMP),
- torch-cpu, torch-cuda: the same algorithm in PyTorch, the dot products by the product of
  the CSR tensor and the dense centroids, the sums of the clusters by index_add_.
"""

import argparse
import json
import time

import numpy as np
import scipy.io
import scipy.sparse


def initial_centroids(X, clusters):
    n = X.shape[0]
    return X[[i * n // clusters for i in range(clusters)]].toarray()


def run_sklearn(X, C0, iterations):
    import sklearn.cluster

    kmeans = sklearn.cluster.KMeans(
        n_clusters=C0.shape[0], init=C0, n_init=1, max_iter=iterations, tol=0, algorithm="lloyd"
    )
    start = time.perf_counter()
    kmeans.fit(X)
    elapsed = time.perf_counter() - start
    # scikit-learn stops when the labels do not change
    return elapsed, kmeans.n_iter_, kmeans.inertia_


def run_torch(X, C0, iterations, device):
    import torch

    dtype = torch.float32 if X.dtype == np.float32 else torch.float64
    indptr = torch.tensor(X.indptr, dtype=torch.int64, device=device)
    indices = torch.tensor(X.indices, dtype=torch.int64, device=device)
    values = torch.tensor(X.data, dtype=dtype, device=device)
    Xt = torch.sparse_csr_tensor(indptr, indices, values, size=X.shape)
    rows = torch.repeat_interleave(torch.arange(X.shape[0], device=device), indptr[1:] - indptr[:-1])
    norms = torch.tensor(np.asarray(X.multiply(X).sum(axis=1)).ravel(), dtype=dtype, device=device)
    K, d = C0.shape

    def assign(C):
        dist = (C * C).sum(dim=1)[None, :] - 2 * (Xt @ C.T)
        best, labels = dist.min(dim=1)
        return labels, (best + norms).sum()

    def update(C, labels):
        counts = torch.bincount(labels, minlength=K)
        sums = torch.zeros(K * d, dtype=dtype, device=device)
        sums.index_add_(0, labels[rows] * d + indices, values)
        sums = sums.view(K, d)
        nonempty = counts > 0
        C = C.clone()
        C[nonempty] = sums[nonempty] / counts[nonempty, None].to(dtype)
        return C

    def lloyd():
        C = torch.tensor(C0, dtype=dtype, device=device)
        for _ in range(iterations):
            labels, _ = assign(C)
            C = update(C, labels)
        return C

    lloyd()  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    C = lloyd()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    _, inertia = assign(C)
    return elapsed, iterations, float(inertia)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--clusters", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--precision", choices=["float", "double"], default="float")
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["sklearn", "torch-cpu", "torch-cuda"],
        choices=["sklearn", "torch-cpu", "torch-cuda"],
    )
    parser.add_argument("--loops", type=int, default=3, help="the minimal time of this number of runs is logged")
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    dtype = np.float32 if args.precision == "float" else np.float64
    X = scipy.sparse.csr_matrix(scipy.io.mmread(args.input_file), dtype=dtype)
    X.sort_indices()

    with open(args.log_file, "a") as log:
        for clusters in args.clusters:
            C0 = initial_centroids(X, clusters).astype(dtype)
            for implementation in args.implementations:
                runs = []
                for _ in range(args.loops):
                    if implementation == "sklearn":
                        runs.append(run_sklearn(X, C0, args.iterations))
                    else:
                        runs.append(run_torch(X, C0, args.iterations, implementation.split("-")[1]))
                elapsed, iterations, inertia = min(runs)
                record = {
                    "operation": "k-means",
                    "matrix": args.input_file,
                    "points": str(X.shape[0]),
                    "features": str(X.shape[1]),
                    "nonzeros": str(X.nnz),
                    "precision": args.precision,
                    "clusters": str(clusters),
                    "iterations": str(args.iterations),
                    "assignment": "",
                    "update": "",
                    "device": "cuda" if implementation == "torch-cuda" else "host",
                    "performer": implementation,
                    "time": f"{elapsed:e}",
                    "loops": str(args.loops),
                    "time per iteration": f"{elapsed / iterations:e}",
                    "inertia": f"{inertia:e}",
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(implementation, clusters, f"{elapsed / iterations * 1e3:.3f} ms/iteration", f"inertia {inertia:.6e}")


if __name__ == "__main__":
    main()
