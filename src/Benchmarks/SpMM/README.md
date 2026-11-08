# SpMM Benchmark

Benchmark of the product of a sparse matrix and a dense matrix,
`C = A * B`, where `A` is an `m x n` sparse matrix in the CSR format and
`B` is an `n x k` dense matrix.
The benchmark compares hand-written reference kernels with vendor libraries
for both narrow (`k` = 1 to 16) and wide (`k` = 32 and more) dense matrices.
The reference kernels will later serve as the performance baseline for the
generic SpMM implementation based on TNL segments.

## Implementations

### Host

- **RowWise** – rows of `A` are distributed among OpenMP threads, every
  thread computes the whole row of `C`. It is also used to compute the
  reference result for the verification.
- **TNL SpMM** – `TNL::Matrices::sparseDenseMatrixProduct` with `A` in the CSR
  format on the host (with OpenMP), with the default configuration and with
  tiles of 8, 32 and 64 columns (`tile T`).
- **TNL SpMM Sequential** – the same with the default configuration on
  `TNL::Devices::Sequential`.
- **Eigen** – the product of `Eigen::Map` of a row-major `Eigen::SparseMatrix`
  and a dense matrix, parallelized over the rows with OpenMP by Eigen
  (when Eigen is found by CMake).
- **Ginkgo** – `gko::matrix::Csr::apply` on `gko::matrix::Dense` with the
  OpenMP executor and the SpMV strategies `automatic`, `classical`,
  `load_balance` and `merge_path` (row-major layout only, when Ginkgo is found
  by CMake).

### GPU (CUDA and HIP)

- **TNL k-SpMV** – `k` calls of `SparseMatrix::vectorProduct` on the columns
  of `B` (column-major layout only). This is the baseline that the fused
  kernels must beat.
- **TNL SpMM** – the generic `TNL::Matrices::sparseDenseMatrixProduct`, which
  computes each tile of at most 16 columns of `C` by one multireduction in the
  segments of `A` (`TNL::Algorithms::Segments::multiReduceAllSegments`). It is
  measured with `A` in the CSR format (`TNL SpMM`) and with all launch
  configurations of its segments (Uniform with `TPS` 1–128 and Tiered).
  With the `Default` configuration, row-major `B` and more than 4 columns,
  it uses `sparseDenseMatrixProductLanesOverColumns` with the numbers of lanes
  and columns per lane selected by `k` and the precision. Otherwise, the `Default` configuration
  is selected by the average row length (Tiered for less than 16 elements,
  Uniform with 16 threads per row otherwise).
  Selected by `--segments-formats`, it is also measured with `A` in other
  formats:
  - `TNL SpMM AdaptiveCSR` – the native adaptive kernel only, with tiles of at
    most 32 bytes (8 columns in single precision), since the adaptive kernel
    does not support larger values.
  - `TNL SpMM Ellpack` and `TNL SpMM SlicedEllpack` – all launch
    configurations of the segments. A format is skipped when it would have more
    elements than `--max-padding-ratio` times the number of nonzeros.
- **TNL SpMM LanesOverColumns** – `TNL::Matrices::sparseDenseMatrixProductLanesOverColumns`
  with `A` in the CSR format (row-major layout only): a group of `G` threads
  processes one row of `A` and every thread computes `V` consecutive columns of
  `C`, loaded by vectorized loads when possible. The very long rows are split
  into pieces processed by whole thread blocks. It is measured with the same
  configurations as the LanesOverColumns reference kernels.
- **RowGroup** – a group of `TPR` threads processes one row of `A` and every
  thread accumulates a tile of `K` columns of `C` in registers. Every element
  of `A` is loaded once per tile.
  - `K == k` is the fused variant of SpMV for small `k`.
  - `K < k` processes the columns in tiles (grid dimension y).
  - Loads of `B` are vectorized in the row-major layout.
  - Works with both row-major and column-major `B` and `C`.
  - In the column-major layout, the tiles of `K` = 1, 2, 4, 8 columns are also
    processed by consecutive kernel launches (`T=1` in `kernel config`), so
    that the columns of `B` used by one launch fit better in the L2 cache.
- **LanesOverColumns** – a group of `G` lanes processes one row of `A`. The
  lanes load chunks of `G` elements of the row cooperatively and broadcast
  them by warp shuffles. Every lane computes `V` consecutive columns of `C`
  with vectorized loads of `B`. This follows GE-SpMM and Sputnik, and it works
  with row-major `B` and `C` only.
- **NnzSplit** – like `LanesOverColumns`, but every group of `G` lanes
  processes a chunk of `L` nonzero elements regardless of the row
  boundaries. The rows split between several chunks are accumulated by atomic
  additions. This balances the work for matrices with very irregular row
  lengths (a simplified merge-based SpMM). Row-major `B` and `C` only.
- **Binned** – the rows are sorted by their lengths and split into bins
  processed by `RowGroup` launches with `TPR` = 1, 2, 4, ..., 32 matching the
  row lengths (like the binning of ACSR). Every thread processes about `E`
  elements of a row. Rows longer than `P` elements are split into pieces of
  `P` elements processed by whole warps with atomic additions. Works with
  both row-major and column-major `B` and `C`. The preprocessing (sorting and
  binning) is not included in the measured time.
- **cuSPARSE** – `cusparseSpMM` with the algorithms `ALG_DEFAULT`,
  `CSR_ALG1`, `CSR_ALG2` and `CSR_ALG3` (CUDA build only).
- **hipSPARSE** – `hipsparseSpMM` with the same algorithms (HIP build only,
  when hipSPARSE is found).

The kernel parameters (`TPR`, `K`, `G`, `V`, `L`, `E`, `P`, or the library algorithm)
are logged in the `kernel config` column. Algorithms not supported by a library
for the given layout are reported as errors and skipped.

The `RowGroup` and `LanesOverColumns` kernels can process the rows either in
the natural order or sorted by their lengths in descending order (row swizzle
of Sputnik), see `--row-order` and `--sorting-window`. The order is logged in
the `row order` column. Sorting makes the rows processed by one warp similarly
long, but it can break the locality of the accesses to `B`; sorting within
windows of consecutive rows keeps it.

The kernels are described in detail, including the references to the
literature, in `ReferenceKernels.h`.

## Measured quantities

- `time`, `bandwidth` – the bandwidth is computed from the minimal memory
  traffic, i.e. reading `A` and `B` once and writing `C` once.
- `GFLOPS` – `2 * nnz * k` floating point operations per multiplication.
- `rel. error` – the max-norm of the difference from the reference result
  divided by the max-norm of the reference result.

## Input matrices

The sparse matrix is either read from a Matrix Market file (`--input-file`)
or generated (`--matrix-generator`):

| Generator | Description | Nonzeros per row |
|-----------|-------------|------------------|
| `laplace-2d` | 5-point stencil on a `grid-size-2d`² grid | 5 |
| `laplace-3d` | 7-point stencil on a `grid-size-3d`³ grid | 7 |
| `stencil-27` | 27-point stencil on a `grid-size-3d`³ grid | 27 |

The generated matrices cover different numbers of nonzeros per row, which
determines where the traffic of `A` stops dominating the traffic of `B` and
`C` (see `plan.md`).

## Usage

```bash
ninja -C build/cuda tnl-benchmark-spmm-cuda
build/cuda/bin/tnl-benchmark-spmm-cuda --device cuda --precision double \
   --dense-columns 1 2 4 8 16 32 64 128 --dense-layout all \
   --log-file spmm.log
```

Important parameters:

- `--input-file` – Matrix Market file. When it is empty, the matrices are generated.
- `--matrix-generator` – `laplace-2d`, `laplace-3d`, `stencil-27` or `all`.
- `--grid-size-2d`, `--grid-size-3d` – grid size of the generated matrices.
- `--dense-columns` – list of the numbers of columns `k` of the dense matrix.
- `--dense-layout` – `row-major`, `column-major` or `all`.
- `--kernels` – list of the benchmarked implementations: `k-spmv`, `segments`,
  `row-group`, `lanes-over-columns`, `nnz-split`, `binned`, `vendor` or `all`.
- `--segments-formats` – formats of `A` for the TNL SpMM: `csr`, `adaptive-csr`,
  `ellpack`, `sliced-ellpack` or `all`.
- `--max-padding-ratio` – the maximal ratio of the number of elements of
  Ellpack and SlicedEllpack including the padding to the number of nonzeros.
- `--row-order` – `natural`, `sorted` or `all`.
- `--sorting-window` – number of consecutive rows sorted together for the
  `sorted` row order, `0` means all rows.
- `--precision` – `float`, `double` or `all`.
- `--device` – `host`, `cuda`, `hip` or `all`.
