# Dense Linear Solvers Benchmark

This benchmark measures the performance of dense direct solvers for linear systems $A x = b$ with dense matrices.
The system matrix is either randomly generated with a given size, or read from a file in the MatrixMarket format.
In both cases, the right-hand side is computed as $b = A x$ with the exact solution $x = (1, \ldots, 1)^T$, so the
benchmark can verify the correctness of the computed solution.

The following solvers are measured:

1. **GEM**, the TNL implementation of the Gaussian elimination method, optionally with pivoting, run on the host
   CPU and, in CUDA or HIP builds, on the GPU.
2. **CuSolverWrapper**, a wrapper around the NVIDIA cuSOLVER library (`cusolverDn`), run on the GPU in CUDA builds.
   In HIP builds the CuSolverWrapper measurement is compiled in as well, but it fails at runtime since cuSOLVER is
   CUDA-only, and the failure is recorded in the log.

The benchmark supports single or double precision, and both precisions can be measured one after the other with
the `--precision all` option.

## Executables

The benchmark is built as a single executable, `tnl-benchmark-dense-linear-solvers`, with the backend selected at
compile time:

| Build | Sources | Linked libraries |
|-------|---------|------------------|
| Host only | `tnl-benchmark-dense-linear-solvers.cpp` | `TNL::TNL` |
| CUDA | `tnl-benchmark-dense-linear-solvers.cu` | `TNL::TNL`, `CUDA::cusolver` |
| HIP | `tnl-benchmark-dense-linear-solvers.hip` | `TNL::TNL` |

## Usage

Build the benchmark with `just` and run it:

```bash
just build tnl-benchmark-dense-linear-solvers
./build/bin/tnl-benchmark-dense-linear-solvers
```

Run with a specific matrix size, in double precision, on the GPU:

```bash
./build/bin/tnl-benchmark-dense-linear-solvers --matrix-size 2048 --precision double --device cuda
```

## Options

### Dense linear solvers benchmark settings

| Option | Description | Default |
|--------|-------------|---------|
| `--matrix-size <n>` | Size of the randomly generated matrix | `128` |
| `--input-file <file>` | Input matrix file name (overrides random matrix generation) | |
| `--pivoting <bool>` | Use pivoting in GEM/LU computation | `true` |
| `--precision <type>` | Precision of the arithmetics (`float`, `double`, `all`) | `double` |
| `--device <device>` | Device to run benchmarks on (`host`, `cuda`, `hip`, `all`) | `all` |

### General benchmark settings

| Option | Description | Default |
|--------|-------------|---------|
| `--log-file <file>` | Log file name for JSONL output | `<program>.log` |
| `--output-mode <mode>` | Mode for opening the log file (`overwrite`, `append`) | `overwrite` |
| `--loops <n>` | Number of iterations for every computation | `10` |
| `--min-time <t>` | Minimal real time in seconds for every computation | `0.0` |
| `--warmup-loops <n>` | Number of warmup iterations before timing (0 to disable) | `1` |
| `--warmup-min-time <t>` | Minimal real time in seconds for warmup (0 to disable) | `0.0` |
| `--verbose <n>` | Verbose mode for terminal output, the higher number the more verbosity | `1` |
| `--catch-exceptions <bool>` | Catch exceptions during timing | `true` |

### Device settings

| Option | Description | Default |
|--------|-------------|---------|
| `--openmp-enabled <bool>` | Enable support of OpenMP (host builds) | `true` |
| `--openmp-max-threads <n>` | Set maximum number of OpenMP threads (host builds) | `omp_get_max_threads()` |
| `--cuda-device <n>` | Choose CUDA device to run the computation (CUDA builds) | `0` |
| `--hip-device <n>` | Choose HIP device to run the computation (HIP builds) | `0` |
