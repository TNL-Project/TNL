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
  (distance, cluster) of the point by an atomic minimum of 64-bit integers (only when the key
  is smaller than the current one, which avoids most atomic operations). Hence the nearest
  centroid is found by the storer and the `n x K` matrix of the distances is never stored
  (`fused` assignment). The `unfused` assignment stores the dot products in a dense matrix by
  `sparseDenseMatrixProduct` and finds the minima in its rows.
- `fused-half` (only on GPUs) reads the centroids from a copy of `C^T` in half precision, which
  is made before every assignment (`DirectFetch` with a transformation to `float`, the
  centroids are loaded by vectorized loads of their bytes). It halves the data read from the
  cache, which limits the assignment for many clusters. The distances are rounded, so the
  points with two nearly equally distant centroids may be assigned differently and the
  inertia differs from `fused` (by about `2e-4` relatively after 10 iterations).

The update computes the sums of the points of the clusters:

- `spmm`: the sums are the product `X^T * Z` of the transposed data matrix and the indicator
  matrix of the clusters, which is given by the lambda function
  `fetchB( point, cluster ) = label( point ) == cluster`, so it is not stored. The storer
  divides the sums by the sizes of the clusters. No atomic operations are needed.
- `atomic`: the elements of the points are added to the sums of their clusters by atomic
  additions.
- `histogram`: the sums are computed by the rows of `X^T`, i.e. separately for every feature,
  so that only `nnz` elements are added (`spmm` computes `nnz * K` products, most of them with
  zeros) and the sums of one feature are written once. On GPUs, the rows are split into pieces
  of at most 256 elements, every piece is processed by a group of 32 threads, which add its
  elements to the `K` sums of its clusters in shared memory. The pieces of the long rows (the
  frequent features) are then added to the sums by atomic additions. The labels are read as
  16-bit integers, so that more of them fit in the L2 cache, and only `C^T` is updated (`C` is
  computed from it when needed). On the host, every row is processed by one thread.

Two optimizations of the fused assignment adapt to the data:

- Tiles of the clusters and the configuration of the product: the product is computed for
  tiles of `T` clusters (`T` columns of `C^T`) one after another. Narrower tiles read `X` more
  times, but the rows of the tile of `C^T` fit better in the cache. On GPUs, every row of `X` is
  processed by `L` threads (lanes), each computing `V` consecutive columns
  (`sparseDenseMatrixProductWithLambdasLanesOverColumns`). The best configuration depends on
  the data, so `SparseKMeans::tuneAssignment` measures the assignment before the benchmark:
  first with `T = K`, 64 and 32 and the default configuration of the product, then with the
  selected `T` and `L * V` from `T / 4` to `T` (`--assignment-tile 0`, the default). On GPUs,
  the configuration is selected separately for the points sorted by their clusters (see below).
  The selected configurations and the time of the tuning are logged in the columns
  `assignment configuration`, `sorted assignment configuration` and `tuning time`, the tuning
  is not included in the measured time.
- Sorting the points by their clusters (`--reordering-period`): the points of one cluster share
  their features, so the rows of `C^T` read for the consecutive rows of the sorted copy of `X`
  are more likely in the cache. The assignment uses the sorted copy, the update and the results
  use the original order. The points are sorted before the iterations 1, 1 + p, 1 + 2p, ... for
  the period `p > 0`. With `p = -1` (the default), they are sorted before every iteration from
  the iteration 1 only when it pays off (only on GPUs): `tuneAssignment` compares the sorting
  followed by the assignment with the assignment of the unsorted points. Without the tuning, the
  points are sorted before the iteration 1 and the assignment including the sorting is compared
  with the assignment of the iteration 0.

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
