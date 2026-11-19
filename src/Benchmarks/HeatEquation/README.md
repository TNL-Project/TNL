# Heat Equation Benchmark

This benchmark measures the performance of solvers for the heat equation

$$
\frac{\partial u}{\partial t} = \Delta u
$$

in one, two, or three dimensions on the domain
$\langle -L_x/2, L_x/2 \rangle \times \langle -L_y/2, L_y/2 \rangle \times \langle -L_z/2, L_z/2 \rangle$,
where $L_x$, $L_y$, and $L_z$ are the domain sizes along the axes.
The boundary conditions are homogeneous Dirichlet, i.e. $u = 0$ on the boundary.
The initial condition is the Gaussian function

$$
u_0(x, y, z) = \delta \exp\left( -\left( \frac{x^2}{\alpha} + \frac{y^2}{\beta} + \frac{z^2}{\gamma} \right) \right)
$$

(in 1D and 2D without the terms of the missing axes).
In the whole space, the solution of the heat equation with this initial condition is

$$
u(x, y, z, t) = \delta \prod_{i} \sqrt{\frac{a_i}{a_i + 4t}} \exp\left( -\frac{x_i^2}{a_i + 4t} \right),
\qquad a_i \in \{ \alpha, \beta, \gamma \},
$$

which approximates the solution with the Dirichlet boundary conditions as long as it is negligible on the boundary
(with the default parameters, it is below $10^{-8}$ on the boundary at the default final time 0.01).

The equation is discretized explicitly in time on a regular grid,
either by the finite difference method (FDM) or by the finite volume method (FVM).
The benchmark measures the wall time of the full time integration.

## Solver implementations

The benchmark compares several implementations of the same numerical scheme.
They differ in how the grid is represented and how the grid points are traversed:

1. `parallel-for` — the state is stored in raw `TNL::Containers::Vector`
   and the grid points are iterated directly with `TNL::Algorithms::parallelFor`.
2. `nd-array` — the state is stored in `TNL::Containers::NDArray` (FDM only).
3. `grid` — the TNL structured grid `TNL::Meshes::Grid` with its entity traversals.

The benchmark runs each implementation for a series of grid sizes:
the size along each axis starts at the minimum dimension
and each following size is the step factor times the previous one, up to the maximum dimension.
Each combination of the sizes is measured separately.
The results include the precision, the scheme, the dimension, the grid sizes, and the implementation as metadata,
and the dataset size is set to the size of one state vector in bytes.

## Executables

One executable is built for each enabled backend, all from the same source
`tnl-benchmark-heat-equation.h`:

| Executable | Backend |
| ---------- | ------- |
| `tnl-benchmark-heat-equation` | host |
| `tnl-benchmark-heat-equation-cuda` | CUDA |
| `tnl-benchmark-heat-equation-hip` | HIP |

## Usage

The benchmark can be built and executed using the following commands:

```bash
just build tnl-benchmark-heat-equation
./build/bin/tnl-benchmark-heat-equation
```

A complete list of all setup parameters can be obtained with:

```bash
./build/bin/tnl-benchmark-heat-equation --help
```

## Command-line options

Options specific to the heat equation benchmark are:

| Option | Description | Default |
|--------|-------------|---------|
| `--implementation` | Implementation of the solver: `parallel-for`, `nd-array`, or `grid` | `grid` |
| `--scheme` | Numerical scheme: `fdm` or `fvm` | `fdm` |
| `--dimension` | Dimension of the problem: `1`, `2`, or `3` | `2` |
| `--device` | Device the computation will run on: `sequential`, `host`, `cuda`, `hip`, or `all` | `all` |
| `--precision` | Precision of the arithmetics: `float`, `double`, or `all` | `double` |
| `--min-x-dimension` | Minimum dimension over the x axis used in the benchmark | `100` |
| `--max-x-dimension` | Maximum dimension over the x axis used in the benchmark | `200` |
| `--x-size-step-factor` | Factor determining how the dimension grows over the x axis | `2` |
| `--min-y-dimension`, `--max-y-dimension`, `--y-size-step-factor` | The same for the y axis | `100`, `200`, `2` |
| `--min-z-dimension`, `--max-z-dimension`, `--z-size-step-factor` | The same for the z axis | `100`, `200`, `2` |
| `--write-data` | Write initial condition and final state to a file | `false` |
| `--domain-x-size`, `--domain-y-size`, `--domain-z-size` | Domain sizes along the axes | `2.0` |
| `--alpha`, `--beta`, `--gamma` | Widths of the Gaussian initial condition along the axes | `0.02` |
| `--delta` | Amplitude of the initial condition | `0.25` |
| `--time-step` | Time step; when set to `0`, it is chosen according to the space steps | `0.0` |
| `--final-time` | Final time of the simulation | `0.01` |
| `--max-iterations` | Maximum time iterations (`0` means no limit) | `0` |
| `--check-solution` | Compute the maximal difference from the exact solution and log it as `max error` | `false` |

Note that the step factors of the axes used by the given dimension must be greater than 1.
The values `cuda` and `hip` of `--device` are effective only in binaries built with CUDA or HIP support;
the device options `host` and `sequential` are always available.

In addition, the benchmark accepts the common options of the `TNL::Benchmarks::Benchmark` framework
(such as `--loops`, `--log-file`, or `--verbose`)
and the device setup options of `TNL::Devices::Host` and `TNL::Devices::GPU`.
See the `--help` output for the full list.

With `--write-data true`, the initial condition and the final state are written to Gnuplot files.
With `--check-solution true`, each problem is solved once more before the measurement
(from the initial condition, independently of the number of loops of the measurement)
and the maximal difference of the result from the exact solution in the whole space is added to the log
as the metadata column `max error`.

## Running and processing results

The scripts `run-tnl-benchmark-heat-equation-1d`, `run-tnl-benchmark-heat-equation-2d`,
and `run-tnl-benchmark-heat-equation-3d` run the benchmark for all implementations, schemes, and precisions
and write one JSON-lines log file per run, named
`tnl-benchmark-heat-equation-<implementation>-<device>-<precision>-<scheme>-<dimension>d.json`.

The script `process-tnl-benchmark-heat-equation.py`, executed in the directory with the log files,
produces HTML tables `tnl-benchmark-heat-equation-<scheme>-<precision>-<dimension>d.html`
with the measured times grouped by implementation and device,
including the speed-up of each implementation over `parallel-for` and the speed-up of the CUDA device over the host.
The script requires `pandas`.
