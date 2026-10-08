# Dense Matrices Benchmark

Performance benchmarks for operations with dense matrices in TNL. The directory contains three benchmarks:
dense matrix multiplication, dense matrix transposition, and dense matrix-vector product. Each benchmark compares
TNL's `DenseMatrix` implementation with external libraries (BLAS, cuBLAS, hipBLAS, MAGMA, CUTLASS, depending on the
build and the availability of the library) and with legacy hand-written GPU kernels from the `Legacy/` directory.
All benchmarks support single (`float`) and double (`double`) precision.

The measured results are written to a log file in the JSONL format. Where a reference implementation is available,
the log also contains `Diff.Max vs <ref>` and `Diff.L2 vs <ref>` columns with the maximum and L2 norms of the
difference between the computed result and the reference result (see `DenseMatricesResult.h`).

## Benchmarks

### Dense matrix multiplication

Measures the time of the dense matrix product `C = A * B`:

- On the host, TNL's `getMatrixProduct` is compared with a BLAS implementation (if BLAS was found at configure time).
- On CUDA devices, the benchmark measures cuBLAS, MAGMA and CUTLASS (each only if the library was found at configure
  time), several legacy kernels (`Kernel 1.1` to `Kernel 1.6`, plus `TensorCores` when compiled with the
  `USE_TENSOR_CORES` CMake option), and TNL. In addition to the standard product, products with transposed operands
  are measured as well (labeled with the suffix `A`, `B` or `AB`, for example `cuBLAS A` for `C = A^T * B`).
- On HIP devices, hipBLAS, TNL and the legacy kernels are measured, including the products with
  transposed operands.
- The number of rows of the first matrix, the number of columns of the first matrix and the number of columns of the
  second matrix are each swept over 11 values evenly spaced between the minimum and maximum given by the options
  below.

### Dense matrix transposition

Measures out-of-place transposition `B = A^T` and, for square matrices, also in-place transposition:

- On the host, only TNL's `getTransposition` is measured.
- On the GPU, the benchmark measures MAGMA (CUDA only, if found), two legacy kernels (`Kernel 2.1` and `Kernel 2.2`),
  TNL's `getTransposition` (`Kernel 2.3`) and in-place transposition (`Kernel 2.4`, square matrices only).
- Matrix sizes are swept over a grid of 21 row counts by 21 column counts evenly spaced between the minimum and
  maximum given by the options below.

### Dense matrix-vector product

Measures the GEMV operation `y = A * x`:

- On the host, TNL's `vectorProduct` is compared with BLAS (if found).
- On the GPU, TNL is measured with both column-major and row-major matrix layouts (`TNL CMO` and `TNL RMO`) and with
  cuBLAS or hipBLAS (column-major layout only).
- Matrix dimensions start at 10 by 10 and grow as powers of two until the number of matrix elements exceeds
  20000 * 20000. Sizes that do not fit into memory are skipped.

## Executables

Each benchmark is built in a host-only, CUDA and HIP variant:

| Benchmark | Host | CUDA | HIP |
|-----------|------|------|-----|
| Matrix multiplication | `tnl-benchmark-dense-matrix-multiplication` | `tnl-benchmark-dense-matrix-multiplication-cuda` | `tnl-benchmark-dense-matrix-multiplication-hip` |
| Matrix transposition | `tnl-benchmark-dense-matrix-transposition` | `tnl-benchmark-dense-matrix-transposition-cuda` | `tnl-benchmark-dense-matrix-transposition-hip` |
| Matrix-vector product | `tnl-benchmark-dense-matrix-vector-product` | `tnl-benchmark-dense-matrix-vector-product-cuda` | `tnl-benchmark-dense-matrix-vector-product-hip` |

The host executables are built when `TNL_BUILD_CPP_TARGETS` is enabled, the CUDA variants when `TNL_BUILD_CUDA` is
enabled and the HIP variants when `TNL_BUILD_HIP` is enabled. The multiplication and transposition benchmarks use
the default dense matrix layout, which is row-major on the host and column-major on the GPU.

External libraries used by the executables:

- The CUDA executables link cuBLAS and, if found, MAGMA and CUTLASS; the corresponding algorithms are then included
  in the benchmark. The CMake option `USE_TENSOR_CORES` adds the `TensorCores` kernel to the CUDA multiplication
  benchmark.
- The HIP executables link hipBLAS.
- The host executables link a BLAS implementation (cblas) if the library and its headers are found at configure
  time; otherwise the BLAS algorithms are skipped.

## Usage

### Compilation

```bash
# Build host benchmark
just build tnl-benchmark-dense-matrix-multiplication

# Build CUDA benchmark
just build tnl-benchmark-dense-matrix-transposition-cuda

# Build HIP benchmark
just build tnl-benchmark-dense-matrix-vector-product-hip
```

### Running

```bash
# Run the multiplication benchmark on the GPU with double precision
./build/bin/tnl-benchmark-dense-matrix-multiplication-cuda --device cuda --precision double

# Run the transposition benchmark on the host with float precision
./build/bin/tnl-benchmark-dense-matrix-transposition --device host --precision float

# Run the matrix-vector product benchmark on the GPU with 20 timing loops
./build/bin/tnl-benchmark-dense-matrix-vector-product-cuda --device cuda --loops 20
```

### Options

All three executables accept the general benchmark options listed below, together with the device settings
(`--openmp-enabled`, `--openmp-max-threads` and `--cuda-device` or `--hip-device`).

| Option | Description | Default |
|--------|-------------|---------|
| `--log-file <file>` | Log file name for JSONL output | `<program>.log` |
| `--output-mode <mode>` | Mode for opening the log file (`overwrite` or `append`) | `overwrite` |
| `--loops <n>` | Number of iterations for every computation | `10` |
| `--min-time <t>` | Minimal real time in seconds for every computation | `0.0` |
| `--warmup-loops <n>` | Number of warmup iterations before timing (0 to disable) | `1` |
| `--warmup-min-time <t>` | Minimal real time in seconds for warmup (0 to disable) | `0.0` |
| `--verbose <n>` | Verbose mode for terminal output, higher is more verbose | `1` |
| `--catch-exceptions <bool>` | Catch exceptions during timing | `true` |

The multiplication and transposition benchmarks share the following options (only the defaults of `--max-rows` and
`--max-columns` differ between them):

| Option | Description | Default |
|--------|-------------|---------|
| `--min-rows <n>` | Minimum number of matrix rows | `100` |
| `--max-rows <n>` | Maximum number of matrix rows | `1000` (multiplication), `5000` (transposition) |
| `--min-columns <n>` | Minimum number of matrix columns | `100` |
| `--max-columns <n>` | Maximum number of matrix columns | `1000` (multiplication), `5000` (transposition) |
| `--fill-mode <mode>` | Method to fill matrices (`linear` or `trigonometric`) | `linear` |
| `--include-legacy-kernels <bool>` | Include legacy kernels in the benchmark (GPU only) | `true` |
| `--precision <type>` | Precision of the arithmetics (`float`, `double` or `all`) | `double` |
| `--device <device>` | Device to run benchmarks on (`host`, `cuda`, `hip` or `all`) | `all` |

The matrix-vector product benchmark supports only the `--precision` and `--device` options from the table above,
with the same defaults.

## Visualizing results

The `tnl-dense-matrices-html-table-generator.py` script converts benchmark log files into HTML tables. It accepts
one or more log files in the JSONL format and writes `dense_matrix_multiplication.html` and
`dense_matrix_transposition.html` (only for the modes with matching data in the logs) into the output directory.
The tables show the measured times, speedups with respect to the reference algorithms (cuBLAS, MAGMA, CUTLASS,
BLAS, as present in the log) and the `Diff.Max vs <ref>` and `Diff.L2 vs <ref>` error norms:

```bash
# Generate HTML tables in the current directory
./tnl-dense-matrices-html-table-generator.py tnl-benchmark-dense-matrix-multiplication-cuda.log

# Specify output directory
./tnl-dense-matrices-html-table-generator.py *.log --output-dir ./tables
```

The script requires `pandas` and the `TNL` Python module from [src/Python](../../Python/).
