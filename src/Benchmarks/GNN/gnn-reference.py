#!/usr/bin/env python3
"""Reference implementations of the GNN inference benchmark.

Computes the same two-layer GCN and GraphSAGE models as tnl-benchmark-gnn, with the same
features and weights given by the hash function of GNNLayers.h, and appends the results
to the log file in the format of the TNL benchmarks:

- PyTorch: the models written with torch.sparse.mm on the CSR tensor of the normalized
  adjacency matrix and torch.nn.functional.linear,
- PyG: GCNConv (cached=True) and SAGEConv (mean aggregation) of PyTorch Geometric with
  edge_index (the message passing by gather and scatter),
- PyG sparse: the same layers with the adjacency given by a torch.sparse CSR tensor
  (sparse-dense products).

The row i of the adjacency matrix contains the sources of the edges ending in the node i.
"""

import argparse
import json
import math
import time

import numpy as np
import scipy.io
import scipy.sparse
import torch

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def hash_uniform(seed, i, j):
    """The function hashUniform of GNNLayers.h, i and j are broadcast arrays."""
    with np.errstate(over="ignore"):
        x = (
            np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
            + i.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
            + j.astype(np.uint64) * np.uint64(0x94D049BB133111EB)
        )
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * 2.0**-52 - 1.0


def random_matrix(rows, columns, seed, scale, dtype, chunk=1 << 16):
    out = np.empty((rows, columns), dtype=dtype)
    j = np.arange(columns)[None, :]
    for begin in range(0, rows, chunk):
        i = np.arange(begin, min(rows, begin + chunk))[:, None]
        out[begin : begin + i.shape[0]] = scale * hash_uniform(seed, i, j)
    return out


def layer_parameters(inputs, outputs, seed, dtype):
    glorot = math.sqrt(6.0 / (inputs + outputs))
    return {
        "W": random_matrix(inputs, outputs, 100 + seed, glorot, dtype),
        "W_root": random_matrix(inputs, outputs, 200 + seed, glorot, dtype),
        "b": random_matrix(1, outputs, 300 + seed, 0.1, dtype)[0],
    }


def to_torch_csr(A, dtype, device):
    A = A.tocsr()
    A.sort_indices()
    return torch.sparse_csr_tensor(
        torch.tensor(A.indptr, dtype=torch.int64),
        torch.tensor(A.indices, dtype=torch.int64),
        torch.tensor(A.data, dtype=dtype),
        size=A.shape,
    ).to(device)


def make_pytorch(model, A, layers, dtype, device):
    """The model written directly in PyTorch."""
    n = A.shape[0]
    pattern = A.copy().tocsr()
    pattern.data[:] = 1
    if model == "gcn":
        pattern.setdiag(1)
        d = 1.0 / np.sqrt(np.asarray(pattern.sum(axis=1)).ravel())
        adj = scipy.sparse.diags(d) @ pattern @ scipy.sparse.diags(d)
    else:
        lengths = np.asarray(pattern.sum(axis=1)).ravel()
        adj = scipy.sparse.diags(np.divide(1.0, lengths, out=np.zeros(n), where=lengths > 0)) @ pattern
    adj = to_torch_csr(adj, dtype, device)
    params = [{k: torch.tensor(v, device=device) for k, v in p.items()} for p in layers]

    def forward(x):
        for index, p in enumerate(params):
            out = torch.sparse.mm(adj, x @ p["W"]) + p["b"]
            if model == "sage":
                out = out + x @ p["W_root"]
            x = torch.relu(out) if index == 0 else out
        return x

    return forward


def make_pyg(model, A, layers, dtype, device, sparse):
    """The model of the layers of PyTorch Geometric."""
    import torch_geometric.nn as gnn

    A = A.tocoo()
    convs = []
    for p in layers:
        inputs, outputs = p["W"].shape
        if model == "gcn":
            conv = gnn.GCNConv(inputs, outputs, cached=True)
            conv.lin.weight.data = torch.tensor(p["W"].T.copy())
            conv.bias.data = torch.tensor(p["b"])
        else:
            conv = gnn.SAGEConv(inputs, outputs, aggr="mean")
            conv.lin_l.weight.data = torch.tensor(p["W"].T.copy())
            conv.lin_l.bias.data = torch.tensor(p["b"])
            conv.lin_r.weight.data = torch.tensor(p["W_root"].T.copy())
        convs.append(conv.to(device=device, dtype=dtype).eval())
    if sparse:
        pattern = A.tocsr()
        pattern.data[:] = 1
        graph = to_torch_csr(pattern, dtype, device)
    else:
        # the edges from the sources (columns) to the targets (rows)
        graph = torch.tensor(np.vstack([A.col, A.row]), dtype=torch.int64, device=device)

    def forward(x):
        for index, conv in enumerate(convs):
            x = conv(x, graph)
            if index == 0:
                x = torch.relu(x)
        return x

    return forward


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--models", nargs="+", default=["gcn", "sage"], choices=["gcn", "sage"])
    parser.add_argument("--input-features", type=int, default=128)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--output-features", type=int, default=16)
    parser.add_argument("--precision", choices=["float", "double"], default="float")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"], choices=["cpu", "cuda"])
    parser.add_argument(
        "--implementations", nargs="+", default=["PyTorch", "PyG", "PyG sparse"], choices=["PyTorch", "PyG", "PyG sparse"]
    )
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    dtype_np = np.float32 if args.precision == "float" else np.float64
    dtype = torch.float32 if args.precision == "float" else torch.float64
    A = scipy.sparse.csr_matrix(scipy.io.mmread(args.input_file))
    n = A.shape[0]
    features = random_matrix(n, args.input_features, 1, 1.0, dtype_np)
    sizes = [args.input_features, args.hidden_features, args.output_features]
    layers = [layer_parameters(sizes[l], sizes[l + 1], l, dtype_np) for l in range(2)]

    with open(args.log_file, "a") as log, torch.inference_mode():
        for device in args.devices:
            x = torch.tensor(features, device=device)
            for model in args.models:
                for implementation in args.implementations:
                    try:
                        if implementation == "PyTorch":
                            forward = make_pytorch(model, A, layers, dtype, device)
                        else:
                            forward = make_pyg(model, A, layers, dtype, device, implementation == "PyG sparse")
                        out = forward(x)  # warm-up, it also caches the normalization of GCNConv
                        times = []
                        for _ in range(args.loops):
                            if device == "cuda":
                                torch.cuda.synchronize()
                            start = time.perf_counter()
                            out = forward(x)
                            if device == "cuda":
                                torch.cuda.synchronize()
                            times.append(time.perf_counter() - start)
                        out = out.double()
                        record = {
                            "operation": "GNN inference",
                            "graph": args.input_file,
                            "nodes": str(n),
                            "edges": str(A.nnz),
                            "precision": args.precision,
                            "input features": str(args.input_features),
                            "hidden features": str(args.hidden_features),
                            "output features": str(args.output_features),
                            "model": model,
                            "part": "forward",
                            "device": "cuda" if device == "cuda" else "host",
                            "performer": implementation,
                            "time": f"{min(times):e}",
                            "time_median": f"{np.median(times):e}",
                            "loops": str(args.loops),
                            "output sum": f"{float(out.sum()):e}",
                            "output norm": f"{float(out.norm()):e}",
                        }
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        print(device, model, implementation, f"{min(times) * 1e3:.3f} ms", record["output sum"], record["output norm"])
                    except Exception as e:
                        print(device, model, implementation, "FAILED:", e)
                    finally:
                        if device == "cuda":
                            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
