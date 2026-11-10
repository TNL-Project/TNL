# k-means Benchmark

Benchmark of Lloyd's algorithm of the k-means clustering of sparse data (e.g. TF-IDF of
documents) implemented by the sparse-dense products of TNL. The points are the rows of a
sparse matrix `X` with `n` rows and `d` columns, `K` is the number of clusters.

## Algorithm

One iteration assigns every point to the nearest centroid and replaces every centroid by the
mean of its points (the centroid of an empty cluster is kept). The squared distance is
`|x|^2 - 2 x.c + |c|^2`, so the assignment needs the dot products of all points and centroids,
i.e. the product `X * C^T` with `K` columns. It is computed by
`TNL::Matrices::sparseDenseMatrixProductWithLambdas`:

- `fetchB( feature, cluster )` reads the transposed centroids `C^T` (`d x K`, row-major). It is
  given by `TNL::Matrices::DirectFetch` (the array of `C^T` and the stride `K`), so that the
  centroids are read by vectorized loads (`fused`), or by a lambda function (`fused-lambda`).
- `storeC( point, cluster, dot )` computes the distance and updates the packed key
  (distance, cluster) of the point by an atomic minimum of 64-bit integers. Hence the nearest
  centroid is found by the storer and the `n x K` matrix of the distances is never stored
  (`fused` assignment). The `unfused` assignment stores the dot products in a dense matrix by
  `sparseDenseMatrixProduct` and finds the minima in its rows.

The update computes the sums of the points of the clusters:

- `spmm`: the sums are the product `X^T * Z` of the transposed data matrix and the indicator
  matrix of the clusters, which is given by the lambda function
  `fetchB( point, cluster ) = label( point ) == cluster`, so it is not stored. The storer
  divides the sums by the sizes of the clusters. No atomic operations are needed.
- `atomic`: the elements of the points are added to the sums of their clusters by atomic
  additions.

The squared norms of the centroids are the sums of `K` rows of `d` elements. On GPUs, the
reduction of the rows of a dense matrix would use only `K` threads, so the norms are computed
as the product of one row of `d` ones (a sparse matrix) and the squares of `C^T` given by a
lambda function. The very long row is split among many threads by the sparse-dense product.

The implementation is in `SparseKMeans.h`.

## Reference implementations

`kmeans-reference.py` runs the same algorithm with the same initial centroids and the same
number of iterations and appends the results to the same log file:

- `sklearn` – `sklearn.cluster.KMeans( algorithm="lloyd" )` on the CSR matrix (CPU, OpenMP).
  Note that it relocates the centroids of empty clusters, so its trajectory differs when a
  cluster becomes empty.
- `torch-cpu`, `torch-cuda` – PyTorch: the product of the CSR tensor and the centroids,
  `argmin` and `index_add_`.

The initial centroids are the points with the indexes `i * n / K`.

## Measured quantities

- `time` – the time of all iterations, `time per iteration`.
- `inertia` – the sum of the squared distances of the points to the final centroids, which
  is compared with the reference implementations.

## Input data

`kmeans-datasets.py` prepares the Matrix Market files with the rows normalized to the unit
norm: synthetic clustered data, the TF-IDF of 20 Newsgroups and RCV1 (downloaded by
scikit-learn).

## Usage

```bash
python3 src/Benchmarks/KMeans/kmeans-datasets.py --output-dir data --datasets synthetic rcv1
build/cuda/bin/tnl-benchmark-kmeans-cuda --input-file data/rcv1.mtx --clusters 8 32 128 \
   --iterations 10 --log-file kmeans.log
python3 src/Benchmarks/KMeans/kmeans-reference.py --input-file data/rcv1.mtx \
   --clusters 8 32 128 --iterations 10 --log-file kmeans.log
```

Use the host binary `tnl-benchmark-kmeans` built by the host compiler for the measurements on
the CPU. The host code of the CUDA binary (`--device host`) was 10-30 times slower, probably
because nvcc does not inline the extended lambda functions in the host code.
