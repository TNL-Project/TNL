# BLAS Benchmark

This benchmark measures the performance of basic array and vector operations, i.e. operations
similar to the BLAS level-1 routines. It compares several TNL implementations with reference
implementations from other libraries. The results of the vector operations are verified against
the expected values, so for these operations the benchmark checks correctness as well as speed.

## What is measured

The benchmark consists of three parts:

1. **Array operations** measure raw array manipulation with `TNL::Containers::Array`:
   comparison (`std::memcmp` and `operator==`), copy (`std::memcpy`, `operator=`, and
   `TNL::Backend::memcpy` for device-to-device copies), `setValue`, allocation (`setSize`), and
   deallocation (`reset`). On the host and the device, and in the GPU builds also with pinned host
   memory (`CudaHost`/`HipHost`) and managed memory (`CudaManaged`/`HipManaged`) allocators,
   including host-to-device and device-to-host copies.
2. **Vector operations** measure reductions and in-place operations on
   `TNL::Containers::Vector`: `max`, `min`, `absMax`, `absMin`, `sum`, l1, l2, and l3 norms,
   scalar product, scalar multiplication, and addition of one, two, or three vectors. The scans
   (prefix sums) are measured by the separate scan benchmark (`src/Benchmarks/Algorithms`).
3. **Triad** (GPU builds only) measures the STREAM-triad operation `a[i] = b[i] + scalar * c[i]`
   combined with copying the input data from the host memory to the device and the result back.
   The variants differ in the memory allocation used: pageable, pinned, zero-copy, and unified
   memory.

## Compared implementations

Depending on the operation, the following implementations are measured:

- TNL legacy static functions (`CommonVectorOperations`, `VectorOperations`),
- TNL expression templates, labeled `ET` where they are compared with other TNL implementations,
- TNL `Algorithms::parallelFor` kernels,
- TNL `Containers::linearCombination` expressions,
- C++ standard library algorithms (`std::max_element`, `std::min_element`, `std::reduce`,
  `std::transform_reduce`),
  executed with parallel execution policies when TBB is available (they are then reported on the
  `host` device, otherwise on the `sequential` device),
- reference BLAS routines via the CBLAS interface on the host, when a BLAS library with the CBLAS
  interface is found at build time,
- cuBLAS on CUDA GPUs or hipBLAS on AMD GPUs,
- Thrust host and device algorithms, when Thrust is available at build time. The host algorithms
  are reported on the `host` device when the Thrust host backend is parallel (TBB or OpenMP),
  otherwise on the `sequential` device.

The optional comparisons (reference BLAS, TBB, Thrust) are enabled automatically when the
respective library is found by CMake during configuration.

## Executables

Depending on the build configuration, the following executables are built:

| Executable | Description |
|------------|-------------|
| `tnl-benchmark-blas` | Host (CPU) build |
| `tnl-benchmark-blas-cuda` | CUDA build, linked with cuBLAS |
| `tnl-benchmark-blas-hip` | HIP build, linked with hipBLAS |

The host executable runs only the host implementations, the CUDA and HIP executables run both the
host and the GPU implementations.

## Command-line options

The benchmark-specific options are:

| Option | Description | Default |
|--------|-------------|---------|
| `--precision <type>` | Precision of the arithmetics (`float`, `double`, or `all`) | `double` |
| `--device <device>` | Device to run the benchmarks on (`sequential`, `host`, `cuda`, `hip`, or `all`) | `all` |
| `--min-size <n>` | Minimum size of arrays/vectors used in the benchmark | `100000` |
| `--max-size <n>` | Maximum size of arrays/vectors used in the benchmark | `10000000` |
| `--size-step-factor <n>` | Factor determining the size of arrays/vectors used in the benchmark. First size is `min-size` and each following size is `size-step-factor * previous size`, up to `max-size` | `2` |

The value of `--size-step-factor` must be greater than 1. Note that the step factor applies only
to the vector operations; the array operations and the triad always double the array size between
two measurements.

For each precision, the benchmarks are run for one device after another: first the sequential
ones, then the host ones, and finally the GPU ones. The array operations and the triad have no
sequential variants.

In addition, the common benchmark settings (e.g. `--loops`, `--log-file`, `--verbose`) and the
device settings (`--openmp-enabled`, `--openmp-max-threads` and `--cuda-device` or
`--hip-device`) are available. A complete list can be obtained with:

```bash
./build/bin/tnl-benchmark-blas-cuda --help
```

## Usage

Build the desired variant of the benchmark and run it, e.g.:

```bash
just build tnl-benchmark-blas-cuda

./build/bin/tnl-benchmark-blas-cuda --precision double --min-size 100000 --max-size 10000000
```

The benchmark outputs timing measurements in the JSONL format to the log file
(`<program>.log` by default) and prints a summary table to the terminal.

The log can be converted to an HTML report with tables and plots by the general tool
`src/Tools/tnl-benchmark-to-html.py`:

```bash
src/Tools/tnl-benchmark-to-html.py tnl-benchmark-blas-cuda.log
```
