# ALS Benchmark

Benchmark of alternating least squares (ALS) for recommender systems with implicit feedback
(Hu, Koren and Volinsky, 2008) implemented by the sparse-dense products of TNL. It computes the
same algorithm as `AlternatingLeastSquares` of the [implicit](https://github.com/benfred/implicit)
library with the conjugate gradient solver, which is the reference implementation on CPUs and
GPUs.

## Algorithm

The confidences `c_ui > 0` of the interactions of `m` users and `n` items are the elements of
the sparse matrix `R`, the preferences are `p_ui = 1` for its elements and 0 otherwise, and the
confidences of the other pairs are 1. The factors of the users (`X`, `m x k`) and the items
(`Y`, `n x k`) minimize

```text
sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + lambda (|X|^2 + |Y|^2).
```

One iteration solves the factors of the users with the fixed factors of the items and then vice
versa (with `R^T`). The factor of the user `u` solves

```text
(G + Y^T (C_u - I) Y) x_u = Y^T C_u p_u,   G = Y^T Y + lambda I,
```

which is approximated by 3 steps of the conjugate gradient method starting from the current
factor. The matrices of the systems are never assembled. The systems of all users are solved at
once and every step consists of:

- the sampled dense-dense product (SDDMM): the dot products `t_ui = y_i . p_u` of the search
  directions `P` and the factors of the items for the elements of `R`, which give the weights
  `(c_ui - 1) t_ui` (and `c_ui - (c_ui - 1) t_ui` for the initial residual) stored as the values of
  a copy of `R`. On GPUs, every element is processed by a group of 8 to 32 threads.
- the sparse-dense product of the matrix of the weights and `Y` computed by
  `TNL::Matrices::sparseDenseMatrixProductWithLambdas` with `Y` given by `DirectFetch`. Its
  storer adds `G p_u`, so that it stores `A p_u` (or the residual) directly.
- the update of the factors, the residuals and the search directions of every user (a group of
  32 threads per user on GPUs).

`G` is computed for chunks of the rows of `Y`, which are then summed (one thread per element of
the Gram matrix of a chunk on GPUs). On GPUs, the numbers of lanes and columns per lane of the
sparse-dense products are selected separately for the users and the items by measuring the
products before the benchmark (`SparseALS::tuneProducts`). The selected configurations and the
time of the tuning are logged in the columns `users configuration`, `items configuration` and
`tuning time`, the tuning is not included in the measured time.

The implementation is in `SparseALS.h`.

## Reference implementations

`als-reference.py` runs the ALS of implicit with the same initial factors and the same number
of iterations and appends the results to the same log file:

- `implicit-cpu` – `implicit.cpu.als.AlternatingLeastSquares( use_cg=True )` (Cython, OpenMP),
- `implicit-cuda` – `implicit.gpu.als.AlternatingLeastSquares` (one thread block per user,
  cuBLAS for `Y^T Y`).

Only the iterations are measured, not the conversion of the data and its transfer to the GPU.

The initial factors are `(h - 0.5) / k`, where `h` from `[0, 1)` is given by the hash splitmix64
of the index of the element (`getInitialFactors` in `ALSBenchmark.h`).

## Measured quantities

- `time` – the time of all iterations, `time per iteration`.
- `loss` – the loss of the final factors divided by the sum of the confidences of all pairs, the
  same as `calculate_training_loss` of implicit, which is compared with the reference
  implementations.

## Input data

`als-datasets.py` downloads the datasets used by `implicit.datasets` and stores the confidences
weighted by BM25 (like the examples of implicit) in the Matrix Market format, the users are the
rows: Last.fm 360K (play counts, 359k users, 292k artists, 17.5M elements) and MovieLens 100k,
1M and 20M (ratings of at least 4, the 20M variant has 138k users, 21k movies, 10M elements).
The installation of implicit with GPU support:

```bash
pip install "implicit[gpu]" h5py
```

The wheels of implicit 0.7.3 for CUDA 13 need `rmm-cu13==26.4.0` (newer versions of RMM are not
binary compatible).

## Usage

```bash
python3 src/Benchmarks/ALS/als-datasets.py --output-dir data --datasets lastfm movielens-20m
build/cuda/bin/tnl-benchmark-als-cuda --input-file data/lastfm.mtx --factors 32 64 128 \
   --iterations 10 --log-file als.log
python3 src/Benchmarks/ALS/als-reference.py --input-file data/lastfm.mtx \
   --factors 32 64 128 --iterations 10 --log-file als.log
```

Use the host binary `tnl-benchmark-als` built by the host compiler for the measurements on the
CPU.
