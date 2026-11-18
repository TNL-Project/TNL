# Parallel For Benchmark

This benchmark measures the performance of `TNL::Algorithms::parallelFor`
for one-, two-, and three-dimensional index ranges.
Each test fills a `TNL::Containers::Vector` of integers through the multi-dimensional `parallelFor`,
so the result shows the overhead of the multi-dimensional indexing compared to the one-dimensional loop.

The sizes along each axis start at the minimum size
and each following size is the step factor times the previous one, up to the maximum size.
The results include the dimension and the sizes along the axes as metadata,
the dataset size is the size of the vector in bytes,
and the performer is `parallelFor 1D`, `parallelFor 2D`, or `parallelFor 3D`.

## Executables

One executable is built for each enabled backend, all from the same source
`tnl-benchmark-parallel-for.h`:

| Executable | Backend |
| ---------- | ------- |
| `tnl-benchmark-parallel-for` | host |
| `tnl-benchmark-parallel-for-cuda` | CUDA |
| `tnl-benchmark-parallel-for-hip` | HIP |

## Usage

```bash
just build tnl-benchmark-parallel-for
./build/bin/tnl-benchmark-parallel-for --help
```

## Command-line options

Options specific to the parallel for benchmark are:

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Device the computation will run on: `sequential`, `host`, `cuda`, `hip`, or `all` | `all` |
| `--min-x-size`, `--max-x-size`, `--x-size-step-factor` | Sizes over the x axis | `1`, `2^20`, `2` |
| `--min-y-size`, `--max-y-size`, `--y-size-step-factor` | Sizes over the y axis | `1`, `2^20`, `2` |
| `--min-z-size`, `--max-z-size`, `--z-size-step-factor` | Sizes over the z axis | `1`, `2^20`, `2` |
| `--fix-array-size` | Measure only the index ranges with the given total size (`0` measures all of them) | `2^30` |

In addition, the benchmark accepts the common options of the `TNL::Benchmarks::Benchmark` framework
(such as `--loops`, `--log-file`, or `--verbose`)
and the device setup options of `TNL::Devices::Host` and `TNL::Devices::GPU`.
