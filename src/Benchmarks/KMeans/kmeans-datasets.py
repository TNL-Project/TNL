#!/usr/bin/env python3
"""Prepares sparse datasets for the k-means benchmark in the Matrix Market format.

The points are the rows, all rows are normalized to the unit Euclidean norm.

- synthetic: points of a given number of clusters, every cluster has its own set of
  features from which most elements of its points are taken,
- 20newsgroups: TF-IDF of the 20 Newsgroups documents (scikit-learn download),
- rcv1: TF-IDF of the RCV1 documents (scikit-learn download, about 650 MB).
"""

import argparse
import pathlib

import numpy as np
import scipy.io
import scipy.sparse
import sklearn.preprocessing


def synthetic(points, features, nonzeros_per_row, clusters, seed):
    rng = np.random.default_rng(seed)
    support_size = max(4 * nonzeros_per_row, features // clusters)
    supports = [rng.choice(features, support_size, replace=False) for _ in range(clusters)]
    labels = rng.integers(clusters, size=points)
    rows = np.repeat(np.arange(points), nonzeros_per_row)
    columns = np.empty(points * nonzeros_per_row, dtype=np.int64)
    for c in range(clusters):
        idx = np.flatnonzero(np.repeat(labels == c, nonzeros_per_row))
        own = rng.random(idx.size) < 0.8
        columns[idx[own]] = rng.choice(supports[c], own.sum())
        columns[idx[~own]] = rng.integers(features, size=(~own).sum())
    values = np.abs(rng.normal(1.0, 0.3, rows.size))
    X = scipy.sparse.csr_matrix((values, (rows, columns)), shape=(points, features))
    X.sum_duplicates()
    return X


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=["synthetic"], choices=["synthetic", "20newsgroups", "rcv1"]
    )
    parser.add_argument("--points", type=int, default=500000)
    parser.add_argument("--features", type=int, default=100000)
    parser.add_argument("--nonzeros-per-row", type=int, default=40)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        if name == "synthetic":
            X = synthetic(args.points, args.features, args.nonzeros_per_row, args.clusters, args.seed)
            name = f"synthetic-{args.points}-{args.features}-{args.nonzeros_per_row}"
        elif name == "20newsgroups":
            from sklearn.datasets import fetch_20newsgroups_vectorized

            X = fetch_20newsgroups_vectorized(subset="all").data
        else:
            from sklearn.datasets import fetch_rcv1

            X = fetch_rcv1().data
        X = sklearn.preprocessing.normalize(scipy.sparse.csr_matrix(X, dtype=np.float64))
        X.eliminate_zeros()
        path = args.output_dir / f"{name}.mtx"
        scipy.io.mmwrite(path, X, field="real", symmetry="general")
        print(f"{path}: {X.shape[0]} points, {X.shape[1]} features, {X.nnz} nonzeros")


if __name__ == "__main__":
    main()
