# Mean-Curvature Flow Benchmark

This benchmark measures the performance of solvers for the mean-curvature flow equation

$$
\frac{\partial u}{\partial t} = Q \nabla \cdot \left( \frac{\nabla u}{Q} \right), \qquad Q = \sqrt{1 + |\nabla u|^2}
$$

in one, two, or three dimensions, discretized by the finite difference method
on the structured grid `TNL::Meshes::Grid` with homogeneous Dirichlet boundary conditions.
The same discretization is used in all dimensions, see `MeanCurvatureFlowFDMGrid.h`,
and it is integrated in time by the explicit Euler method.
Each time step consists of two traversals of the grid:
the computation of the gradient norms and the update of the solution.
The TNL implementations launch these kernels on GPUs without waiting for each of them to finish
(see `TNL::Benchmarks::getAsyncLaunchConfiguration`), like the other libraries;
the device is synchronized after the time integration.

The benchmark compares the following implementations of the same scheme:

1. `grid` — the TNL structured grid `TNL::Meshes::Grid` with its entity traversals.
2. `parallel-for` — the state is stored in raw `TNL::Containers::Vector` objects
   and the cells are traversed by the multi-dimensional `TNL::Algorithms::parallelFor`.
3. `kokkos` — the scheme implemented with [Kokkos](https://kokkos.org/).
4. `amrex` — the scheme implemented with [AMReX](https://amrex-codes.github.io/amrex/).
5. `gridtools` — the scheme implemented with [GridTools](https://github.com/GridTools/gridtools).

The implementations `kokkos`, `amrex`, and `gridtools` are built only with the CMake options
`TNL_BENCHMARKS_USE_KOKKOS`, `TNL_BENCHMARKS_USE_AMREX`, and `TNL_BENCHMARKS_USE_GRIDTOOLS`, see the section *Comparison with other libraries*
in the README of the heat equation benchmark for the supported library versions, backends, and devices.

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
run the benchmark for all implementations, precisions and devices and write one JSON-lines log file per run,
named `tnl-benchmark-mean-curvature-flow-<implementation>-<device>-<precision>-<scheme>-<dimension>d.json`.
The devices can be selected by the environment variable `DEVICES` (by default `sequential host cuda`).

## Command-line options

The most important options specific to this benchmark are:

| Option | Description | Default |
|--------|-------------|---------|
| `--implementation` | Implementation of the solver: `grid`, `parallel-for`, `kokkos`, `amrex`, or `gridtools` | `grid` |
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
