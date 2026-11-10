#!/usr/bin/env python3
"""Exports the graphs of the standard GNN datasets of PyTorch Geometric to the Matrix Market format.

The row i of the exported adjacency matrix contains the sources of the edges ending in the
node i (the convention of tnl-benchmark-gnn). The numbers of the features and the classes of
the datasets are printed, they can be passed to the benchmark and to gnn-reference.py.
"""

import argparse
import pathlib

import numpy as np
import scipy.io
import scipy.sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--download-dir", type=pathlib.Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["Cora", "PubMed", "Reddit"], choices=["Cora", "PubMed", "Reddit"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch_geometric.datasets as datasets

    for name in args.datasets:
        if name == "Reddit":
            dataset = datasets.Reddit(str(args.download_dir / name))
        else:
            dataset = datasets.Planetoid(str(args.download_dir), name)
        data = dataset[0]
        source, target = data.edge_index.numpy()
        n = data.num_nodes
        A = scipy.sparse.csr_matrix((np.ones(source.size), (target, source)), shape=(n, n))
        A.sum_duplicates()
        A.data[:] = 1
        path = args.output_dir / f"{name}.mtx"
        scipy.io.mmwrite(path, A, field="pattern", symmetry="general")
        print(f"{path}: {n} nodes, {A.nnz} edges, {dataset.num_features} features, {dataset.num_classes} classes")


if __name__ == "__main__":
    main()
