# Eigen Solvers Benchmarks

These benchmarks measure the performance of TNL's experimental eigensolvers from
`TNL::Solvers::Eigen::experimental`, namely the power iteration, the shifted power
iteration, and the QR algorithm. Despite the directory name, they do not use the
third-party [Eigen](https://eigen.tuxfamily.org/) library; "Eigen" refers to
eigenvalue problems.

Each benchmark runs the solvers for a sequence of decreasing convergence thresholds
(epsilon = 10^-1, 10^-3, ..., down to 10^-7 for `float` or 10^-13 for `double`) and
stops when the solver fails to converge. In addition to the measured time, the result
table reports the average number of iterations and the residual error
(see `EigenBenchmarkResult.h`).

## Benchmarks

### `tnl-benchmark-eigen-pi`

Computes the dominant eigenvalue and eigenvector of a matrix stored in the
[Matrix Market](https://math.nist.gov/MatrixMarket/) format using the power iteration
(PI). When `--shift-value` is nonzero, it also runs the shifted power iteration with
the given shift (SPI) and with zero shift (SPI0). The matrix is processed as a sparse
matrix (SM), and additionally as dense matrices with column-major (DM_CMO) and
row-major (DM_RMO) storage if its size is at most 1600. Runs on the host, and on the
GPU when the benchmark is built with CUDA or HIP support.

### `tnl-benchmark-eigen-qr`

Computes all eigenvalues and eigenvectors of a dense matrix stored in the Matrix
Market format using the QR algorithm with three QR factorization methods: Householder
reflections, Gram-Schmidt orthogonalization, and Givens rotations (dense matrices with
column-major and row-major storage are tested). Runs on the host only, even though
the `--device` option is accepted.

### `tnl-benchmark-eigen-random`

Runs the power iteration and the QR algorithm (with all three factorization methods)
on randomly generated symmetric matrices. Dense matrices are tested for sizes from
`--min-size-dense` up to `--max-size-dense` (doubled in each step), sparse matrices
similarly from `--min-size-sparse` up to `--max-size-sparse`. The power iteration runs
on the host and on the GPU (with CUDA or HIP builds), the QR algorithm on the host
only. Individual operations can be toggled with the `--with-*` options.

## Usage

### Compilation

The directory builds three executables. Each is compiled from the `.cu` sources when
CUDA is enabled, from the `.hip` sources when HIP is enabled, and from the `.cpp`
sources otherwise; the target names are the same in all cases:

```bash
# Build all three benchmarks
just build tnl-benchmark-eigen-pi tnl-benchmark-eigen-qr tnl-benchmark-eigen-random

# Build a single benchmark
just build tnl-benchmark-eigen-pi
```

### Running

```bash
# Power iteration on a matrix file, host only
./build/bin/tnl-benchmark-eigen-pi --input-matrix matrix.mtx --device host

# Power and shifted power iteration with a shift of 10.0
./build/bin/tnl-benchmark-eigen-pi --input-matrix matrix.mtx --shift-value 10.0

# QR algorithm on a matrix file
./build/bin/tnl-benchmark-eigen-qr --input-matrix matrix.mtx

# Random matrices with custom size ranges, power iteration only
./build/bin/tnl-benchmark-eigen-random --max-size-dense 4000 --with-qr-householder false \
   --with-qr-gram-schmidt false --with-qr-givens false
```

## Options

All three benchmarks share the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--log-file <file>` | Log file name for JSONL output | `<program>.log` |
| `--output-mode <mode>` | Log file mode (`overwrite` or `append`) | `overwrite` |
| `--loops <n>` | Number of iterations for every computation | `10` |
| `--min-time <t>` | Minimal real time in seconds for every computation | `0` |
| `--warmup-loops <n>` | Number of warmup iterations before timing | `1` |
| `--warmup-min-time <t>` | Minimal real time in seconds for warmup | `0` |
| `--verbose <n>` | Verbosity of the terminal output | `1` |
| `--catch-exceptions <bool>` | Catch exceptions during timing | `true` |
| `--precision <type>` | Precision of the arithmetics (`float`, `double`, `all`) | see below |
| `--device <device>` | Device to run benchmarks on (`host`, `cuda`, `hip`, `all`) | `all` |

The default of `--precision` is `double` for `tnl-benchmark-eigen-pi` and
`tnl-benchmark-eigen-qr`, and `all` for `tnl-benchmark-eigen-random`.

Options specific to `tnl-benchmark-eigen-pi`:

| Option | Description | Default |
|--------|-------------|---------|
| `--input-matrix <file>` | Path to the input matrix in Matrix Market format (`.mtx`), required | (required) |
| `--shift-value <value>` | Shift value for the shifted power iteration method | `0.0` |

`tnl-benchmark-eigen-qr` adds:

| Option | Description | Default |
|--------|-------------|---------|
| `--input-matrix <file>` | Path to the input matrix in Matrix Market format (`.mtx`), required | (required) |

`tnl-benchmark-eigen-random` adds:

| Option | Description | Default |
|--------|-------------|---------|
| `--min-size-dense <n>` | Minimum dense matrix size | `10` |
| `--max-size-dense <n>` | Maximum dense matrix size | `2000` |
| `--min-size-sparse <n>` | Minimum sparse matrix size | `100` |
| `--max-size-sparse <n>` | Maximum sparse matrix size | `10000` |
| `--with-pi <bool>` | Run power iteration benchmarks | `true` |
| `--with-qr-householder <bool>` | Run QR algorithm with Householder factorization | `true` |
| `--with-qr-gram-schmidt <bool>` | Run QR algorithm with Gram-Schmidt factorization | `true` |
| `--with-qr-givens <bool>` | Run QR algorithm with Givens factorization | `true` |

A complete list of all options, including the device settings, can be obtained with:

```bash
./build/bin/tnl-benchmark-eigen-pi --help
```
