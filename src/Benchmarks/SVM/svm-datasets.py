#!/usr/bin/env python3
"""Prepares multiclass sparse datasets for the SVM gradient benchmark.

The datasets from the LIBSVM collection
(www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets) are stored as the Matrix Market file
NAME.mtx with the samples as rows and the text file NAME.labels with the classes
0, ..., K - 1 of the samples (one per line):

- news20: news20.scale, 15,935 samples, 62,061 features, 20 classes,
- sector: sector.scale, 6,412 samples, 55,197 features, 105 classes,
- rcv1: rcv1_test.multiclass, 518,571 samples, 47,236 features, 53 classes.
"""

import argparse
import bz2
import pathlib
import urllib.request

import numpy as np
import scipy.io
import scipy.sparse
import sklearn.datasets

URL_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
FILES = {
    "news20": "news20.scale.bz2",
    "sector": "sector/sector.scale.bz2",
    "rcv1": "rcv1_test.multiclass.bz2",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=["news20"], choices=list(FILES)
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        archive = args.output_dir / pathlib.Path(FILES[name]).name
        if not archive.is_file():
            print(f"downloading {URL_BASE + FILES[name]}")
            urllib.request.urlretrieve(URL_BASE + FILES[name], archive)
        with bz2.open(archive, "rb") as f:
            X, y = sklearn.datasets.load_svmlight_file(f, dtype=np.float64)
        X = scipy.sparse.csr_matrix(X)
        X.eliminate_zeros()
        X.sort_indices()
        _, labels = np.unique(y, return_inverse=True)
        scipy.io.mmwrite(
            args.output_dir / f"{name}.mtx", X, field="real", symmetry="general"
        )
        np.savetxt(args.output_dir / f"{name}.labels", labels, fmt="%d")
        print(
            f"{name}: {X.shape[0]} samples, {X.shape[1]} features, {X.nnz} nonzeros,",
            f"{labels.max() + 1} classes",
        )


if __name__ == "__main__":
    main()
