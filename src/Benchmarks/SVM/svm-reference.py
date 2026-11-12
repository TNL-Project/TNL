#!/usr/bin/env python3
"""Reference implementations of the SVM gradient benchmark.

Computes the gradient of the multiclass (one-vs-rest) linear SVM with the squared hinge
loss, W + X^T G with G_ic = -2 C y_ic max(0, 1 - y_ic (X W)_ic), with the same initial
weights as tnl-benchmark-svm (see getInitialWeights in SVMBenchmark.h) by PyTorch and
appends the results to the log file in the format of the TNL benchmarks:

- torch-spmm: the products of the CSR tensors X and X^T and the dense matrices with all
  classifiers (cuSPARSE SpMM on GPUs),
- torch-spmv: the products with the columns of the dense matrices, one classifier at
  once (cuSPARSE SpMV on GPUs).

Both run on the CPU (the "-cpu" suffix) and on the GPU ("-cuda"). X^T is transposed
before the measurement, like in TNL. The time is the mean of the given number of runs
after one warm-up run.
"""

import argparse
import json
import time

import numpy as np
import scipy.io
import scipy.sparse


def initial_weights(features, classes):
    """The same as getInitialWeights in SVMBenchmark.h."""
    with np.errstate(over="ignore"):
        z = np.arange(features * classes, dtype=np.uint64) * np.uint64(
            0x9E3779B97F4A7C15
        )
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    h = (z >> np.uint64(40)).astype(np.float64) / float(1 << 24)
    return (0.2 * (h - 0.5)).astype(np.float32).reshape(features, classes)


def to_torch_csr(A, device):
    import torch

    return torch.sparse_csr_tensor(
        torch.tensor(A.indptr, dtype=torch.int64),
        torch.tensor(A.indices, dtype=torch.int64),
        torch.tensor(A.data, dtype=torch.float32),
        size=A.shape,
    ).to(device)


def run_torch(X, labels, classes, cost, method, device, loops):
    import torch

    Xt = to_torch_csr(X, device)
    XTt = to_torch_csr(scipy.sparse.csr_matrix(X.T), device)
    W = torch.tensor(initial_weights(X.shape[1], classes), device=device)
    Y = torch.where(
        torch.tensor(labels, device=device)[:, None]
        == torch.arange(classes, device=device)[None, :],
        1.0,
        -1.0,
    )
    WT = W.T.contiguous()
    YT = Y.T.contiguous()

    def gradient_spmm():
        S = Xt @ W
        margin = 1 - Y * S
        G = torch.where(margin > 0, -2 * cost * Y * margin, 0.0)
        return XTt @ G + W

    def gradient_spmv():
        S = torch.stack([torch.mv(Xt, WT[c]) for c in range(classes)])
        margin = 1 - YT * S
        G = torch.where(margin > 0, -2 * cost * YT * margin, 0.0)
        return torch.stack([torch.mv(XTt, G[c]) for c in range(classes)]) + WT

    compute = gradient_spmm if method == "spmm" else gradient_spmv
    compute()  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(loops):
        grad = compute()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / loops
    return elapsed, float(torch.linalg.norm(grad))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument(
        "--labels-file", help="the input file with the extension .labels by default"
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--cost", type=float, default=1.0)
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=[
            "torch-spmm-cpu",
            "torch-spmv-cpu",
            "torch-spmm-cuda",
            "torch-spmv-cuda",
        ],
        choices=[
            "torch-spmm-cpu",
            "torch-spmv-cpu",
            "torch-spmm-cuda",
            "torch-spmv-cuda",
        ],
    )
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    X = scipy.sparse.csr_matrix(scipy.io.mmread(args.input_file), dtype=np.float32)
    X.sort_indices()
    labels_file = args.labels_file or args.input_file.rsplit(".", 1)[0] + ".labels"
    dataset_labels = np.loadtxt(labels_file, dtype=np.int64)
    dataset_classes = int(dataset_labels.max()) + 1

    with open(args.log_file, "a") as log:
        for requested in args.classes:
            classes = requested if requested > 0 else dataset_classes
            labels = dataset_labels % classes
            for implementation in args.implementations:
                _, method, device = implementation.split("-")
                elapsed, norm = run_torch(
                    X, labels, classes, args.cost, method, device, args.loops
                )
                record = {
                    "operation": "SVM gradient",
                    "matrix": args.input_file,
                    "samples": str(X.shape[0]),
                    "features": str(X.shape[1]),
                    "nonzeros": str(X.nnz),
                    "precision": "float",
                    "classes": str(classes),
                    "method": method,
                    "device": "cuda" if device == "cuda" else "host",
                    "performer": f"torch-{method}",
                    "time": f"{elapsed:e}",
                    "loops": str(args.loops),
                    "gradient norm": f"{norm:e}",
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(
                    implementation,
                    classes,
                    f"{elapsed * 1e3:.3f} ms",
                    f"norm {norm:.6e}",
                )


if __name__ == "__main__":
    main()
