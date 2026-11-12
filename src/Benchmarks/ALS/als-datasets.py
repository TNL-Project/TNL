#!/usr/bin/env python3
"""Prepares implicit feedback datasets for the ALS benchmark (Matrix Market format).

The rows are the users, the columns are the items and the values are the confidences
weighted by BM25 like in the examples of the implicit library. The users and the items
without any interaction are removed.

- lastfm: the play counts of the Last.fm 360K dataset,
- movielens-20m, movielens-1m, movielens-100k: the ratings of at least 4 of the
  MovieLens datasets, used as binary interactions.

The datasets are downloaded from the releases of github.com/benfred/recommender_data
(the files used by implicit.datasets) to the cache directory.
"""

import argparse
import pathlib
import urllib.request

import h5py
import numpy as np
import scipy.io
import scipy.sparse

URL_BASE = "https://github.com/benfred/recommender_data/releases/download/v1.0/"


def bm25_weight(X, K1=100, B=0.8):
    """BM25 weighting of the rows of X like implicit.nearest_neighbours.bm25_weight."""
    X = scipy.sparse.coo_matrix(X)
    N = float(X.shape[0])
    idf = np.log(N) - np.log1p(np.bincount(X.col))
    row_sums = np.ravel(X.sum(axis=1))
    length_norm = (1.0 - B) + B * row_sums / row_sums.mean()
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X


def download(cache_dir, filename):
    path = cache_dir / filename
    if not path.is_file():
        print(f"downloading {URL_BASE + filename}")
        urllib.request.urlretrieve(URL_BASE + filename, path)
    return path


def read_matrix(path, name):
    with h5py.File(path, "r") as f:
        m = f.get(name)
        return scipy.sparse.csr_matrix(
            (m.get("data")[:], m.get("indices")[:], m.get("indptr")[:])
        )


def remove_empty(X):
    X = scipy.sparse.csr_matrix(X)
    X.eliminate_zeros()
    rows = np.flatnonzero(np.diff(X.indptr))
    X = X[rows]
    columns = np.flatnonzero(np.bincount(X.indices, minlength=X.shape[1]))
    return X[:, columns]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        help="directory of the downloaded files (the output directory)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["movielens-100k"],
        choices=["lastfm", "movielens-20m", "movielens-1m", "movielens-100k"],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        if name == "lastfm":
            # artists x users, the weighting of the example of implicit
            plays = read_matrix(
                download(cache_dir, "lastfm_360k.hdf5"), "artist_user_plays"
            )
            X = bm25_weight(plays.astype(np.float64)).T
        else:
            variant = name.split("-")[1]
            # movies x users
            ratings = read_matrix(
                download(cache_dir, f"movielens_{variant}.hdf5"), "movie_user_ratings"
            )
            ratings.data = (ratings.data >= 4.0).astype(np.float64)
            ratings.eliminate_zeros()
            X = bm25_weight(ratings).T
        X = remove_empty(X)
        X.sort_indices()
        path = args.output_dir / f"{name}.mtx"
        scipy.io.mmwrite(path, X, field="real", symmetry="general")
        print(f"{path}: {X.shape[0]} users, {X.shape[1]} items, {X.nnz} nonzeros")


if __name__ == "__main__":
    main()
