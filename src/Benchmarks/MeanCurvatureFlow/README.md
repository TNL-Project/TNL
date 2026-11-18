# Mean-Curvature Flow Benchmark

This benchmark measures the performance of solvers for the mean-curvature flow equation
in one, two, or three dimensions, discretized by the finite difference method
on the structured grid `TNL::Meshes::Grid`.
Two time discretizations are available: the explicit scheme
and the semi-implicit scheme, which solves a linear system in each time step.

The benchmark runs the solver for a series of grid sizes:
the size along each axis starts at the minimum dimension
and each following size is the step factor times the previous one, up to the maximum dimension.
The results include the precision, the scheme, the dimension, the grid sizes, and the implementation as metadata,
and the dataset size is set to the size of one state vector in bytes.

## Executables

One executable is built for each enabled backend, all from the same source
`tnl-benchmark-mean-curvature-flow.h`:

| Executable | Backend |
| ---------- | ------- |
| `tnl-benchmark-mean-curvature-flow` | host |
| `tnl-benchmark-mean-curvature-flow-cuda` | CUDA |
| `tnl-benchmark-mean-curvature-flow-hip` | HIP |

## Usage

```bash
just build tnl-benchmark-mean-curvature-flow
./build/bin/tnl-benchmark-mean-curvature-flow --help
```

The scripts `run-tnl-benchmark-mean-curvature-flow-1d` and `run-tnl-benchmark-mean-curvature-flow-2d`
run the benchmark for a predefined set of parameters.

## Command-line options

The most important options specific to this benchmark are:

| Option | Description | Default |
|--------|-------------|---------|
| `--time-discretisation` | Time discretization: `explicit` or `semi-implicit` | `semi-implicit` |
| `--dimension` | Dimension of the problem: `1`, `2`, or `3` | `2` |
| `--device` | Device the computation will run on: `sequential`, `host`, `cuda`, `hip`, or `all` | `all` |
| `--precision` | Precision of the arithmetics: `float`, `double`, or `all` | `double` |
| `--final-time` | Final time of the simulation | `0.1` |
| `--max-iterations` | Maximum time iterations (`0` means no limit) | `0` |

The grid sizes, the domain sizes, and the parameters of the initial condition are set by the same options
as in the heat equation benchmark, see `--help` for the full list.
In addition, the benchmark accepts the common options of the `TNL::Benchmarks::Benchmark` framework
(such as `--loops`, `--log-file`, or `--verbose`)
and the device setup options of `TNL::Devices::Host` and `TNL::Devices::GPU`.
