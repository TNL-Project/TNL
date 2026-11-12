#!/usr/bin/env python3
"""Reference implementations of the dense-sparse product benchmark.

Computes C = B A for the sparse matrix A (n x m) from the input file and the dense
matrix B (k x n, row-major) with the same elements as tnl-benchmark-dense-sparse (see
getDenseMatrix in DenseSparseBenchmark.h) by PyTorch and appends the results to the log
file in the format of the TNL benchmarks:

- torch: B @ A with the CSR tensor A,
- torch-transpose-dense: (A^T @ B^T)^T with the precomputed CSR tensor A^T and the
  row-major copy of B^T (cuSPARSE SpMM on GPUs), the result is copied to a row-major
  tensor.

Both run on the CPU (the "-cpu" suffix) and on the GPU ("-cuda"). The time is the mean
of the given number of runs after one warm-up run.
"""

import argparse
import json
import time

import numpy as np
import scipy.io
import scipy.sparse


def dense_matrix(k, n):
    """The same as getDenseMatrix in DenseSparseBenchmark.h."""
    with np.errstate(over="ignore"):
        z = np.arange(k * n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return (
        ((z >> np.uint64(40)).astype(np.float64) / float(1 << 24) - 0.5)
        .astype(np.float32)
        .reshape(k, n)
    )


def to_torch_csr(A, device):
    import torch

    return torch.sparse_csr_tensor(
        torch.tensor(A.indptr, dtype=torch.int64),
        torch.tensor(A.indices, dtype=torch.int64),
        torch.tensor(A.data, dtype=torch.float32),
        size=A.shape,
    ).to(device)


def run_torch(A, k, method, device, loops):
    import torch

    At = to_torch_csr(A, device)
    ATt = to_torch_csr(scipy.sparse.csr_matrix(A.T), device)
    B = torch.tensor(dense_matrix(k, A.shape[0]), device=device)

    def product():
        return B @ At

    def transposed_product():
        # the row-major copy of B^T, so that cuSPARSE reads the row-major dense matrix
        return (ATt @ B.t().contiguous()).t().contiguous()

    compute = product if method == "torch" else transposed_product
    compute()  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(loops):
        C = compute()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / loops
    return elapsed, float(C.double().sum())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument(
        "--dense-rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=[
            "torch-cpu",
            "torch-transpose-dense-cpu",
            "torch-cuda",
            "torch-transpose-dense-cuda",
        ],
        choices=[
            "torch-cpu",
            "torch-transpose-dense-cpu",
            "torch-cuda",
            "torch-transpose-dense-cuda",
        ],
    )
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    A = scipy.sparse.csr_matrix(scipy.io.mmread(args.input_file), dtype=np.float32)
    A.sort_indices()

    with open(args.log_file, "a") as log:
        for k in args.dense_rows:
            for implementation in args.implementations:
                method, device = implementation.rsplit("-", 1)
                elapsed, checksum = run_torch(A, k, method, device, args.loops)
                record = {
                    "operation": "dense-sparse product",
                    "matrix": args.input_file,
                    "rows": str(A.shape[0]),
                    "columns": str(A.shape[1]),
                    "nonzeros": str(A.nnz),
                    "precision": "float",
                    "dense rows": str(k),
                    "organization": "row-major",
                    "method": "",
                    "device": "cuda" if device == "cuda" else "host",
                    "performer": method,
                    "time": f"{elapsed:e}",
                    "loops": str(args.loops),
                    "checksum": f"{checksum:e}",
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(
                    implementation,
                    k,
                    f"{elapsed * 1e3:.3f} ms",
                    f"checksum {checksum:.6e}",
                )


if __name__ == "__main__":
    main()
