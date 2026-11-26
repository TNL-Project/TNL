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

The benchmark is built once per backend, all from the same source `tnl-benchmark-dense-linear-solvers.h`:

| Executable | Linked libraries |
|------------|------------------|
| `tnl-benchmark-dense-linear-solvers` (host) | `TNL::TNL` |
| `tnl-benchmark-dense-linear-solvers-cuda` | `TNL::TNL`, `CUDA::cusolver` |
| `tnl-benchmark-dense-linear-solvers-hip` | `TNL::TNL` |

## Usage

Build the benchmark with `just` and run it:

```bash
just build tnl-benchmark-dense-linear-solvers tnl-benchmark-dense-linear-solvers-cuda
./build/bin/tnl-benchmark-dense-linear-solvers
./build/bin/tnl-benchmark-dense-linear-solvers-cuda --device cuda
```

Run with a specific matrix size, in double precision, on the GPU:

```bash
./build/bin/tnl-benchmark-dense-linear-solvers-cuda --matrix-size 2048 --precision double --device cuda
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

### Common settings

The common benchmark options (output, number of iterations and device settings)
are described in the [benchmarks README](../README.md#common-command-line-options).
