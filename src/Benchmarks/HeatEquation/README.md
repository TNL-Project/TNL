# Heat Equation Benchmark

This benchmark measures the performance of solvers for the two-dimensional heat equation

$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}
$$

on a rectangular domain $\langle -L_x/2, L_x/2 \rangle \times \langle -L_y/2, L_y/2 \rangle$, where $L_x$ and $L_y$
are the domain sizes along the x and y axes. The boundary conditions are homogeneous Dirichlet, i.e. $u = 0$ on the
boundary. The initial condition is a bump-shaped function

$$
u(x, y, 0) = \max\left( \frac{x^2}{\alpha} + \frac{y^2}{\beta} + \gamma, 0 \right) \cdot 0.2
$$

with the parameters $\alpha$, $\beta$, and $\gamma$ settable from the command line.

The equation is discretized by the explicit finite difference scheme (forward Euler in time, centered differences in
space) on a regular grid. The time step is either fixed by the user or, by default, chosen as $0.1 \cdot \min(h_x^2, h_y^2)$,
where $h_x$ and $h_y$ are the space steps. The benchmark then measures the wall time of the full time integration.

## Solver implementations

The benchmark compares four implementations of the same numerical scheme. They differ in how the grid is represented
and how the interior grid points are traversed:

1. `parallel-for` — the state is stored in raw `TNL::Containers::Vector` and the interior points are iterated
   directly with `TNL::Algorithms::parallelFor`.
2. `simple-grid` — a minimal hand-written 2D grid and entity abstraction on top of `parallelFor`.
3. `grid` — the TNL structured grid `TNL::Meshes::Grid` with the `forInteriorEntities` traversal.
4. `nd-grid` — a prototype of a general N-dimensional grid with its own traversal mechanism.

The benchmark runs each implementation for a series of grid dimensions: the size along each axis starts at the
minimum dimension and each following size is the step factor times the previous one, up to the maximum dimension.
Each combination of the x and y sizes is measured separately. The results include the precision, the grid dimensions,
and the implementation as metadata, and the dataset size is set to the size of one state vector.

## Executable

The build produces a single executable, `tnl-benchmark-heat-equation`, for all backends. CMake compiles it from
`tnl-benchmark-heat-equation.cpp` for host-only builds, from `tnl-benchmark-heat-equation.cu` for CUDA builds,
and from `tnl-benchmark-heat-equation.hip` for HIP builds.

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
| `--implementation` | Implementation of the heat equation solver: `parallel-for`, `simple-grid`, `grid`, or `nd-grid` | `grid` |
| `--device` | Device the computation will run on: `sequential`, `host`, `cuda`, `hip`, or `all` | `all` |
| `--precision` | Precision of the arithmetics: `float`, `double`, or `all` | `double` |
| `--min-x-dimension` | Minimum dimension over the x axis used in the benchmark | `100` |
| `--max-x-dimension` | Maximum dimension over the x axis used in the benchmark | `200` |
| `--x-size-step-factor` | Factor determining how the dimension grows over the x axis | `2` |
| `--min-y-dimension` | Minimum dimension over the y axis used in the benchmark | `100` |
| `--max-y-dimension` | Maximum dimension over the y axis used in the benchmark | `200` |
| `--y-size-step-factor` | Factor determining how the dimension grows over the y axis | `2` |
| `--write-data` | Write initial condition and final state to a file | `false` |
| `--domain-x-size` | Domain size along the x axis | `2.0` |
| `--domain-y-size` | Domain size along the y axis | `2.0` |
| `--alpha` | Alpha value in the initial condition | `-0.05` |
| `--beta` | Beta value in the initial condition | `-0.05` |
| `--gamma` | Gamma value in the initial condition | `5.0` |
| `--time-step` | Time step; when set to `0`, the time step is chosen as `0.1 * min(h_x^2, h_y^2)` | `0.0` |
| `--final-time` | Final time of the simulation | `0.01` |
| `--max-iterations` | Maximum time iterations (`0` means no limit) | `0` |

Note that the step factors must be greater than 1. The values `cuda` and `hip` of `--device` are effective only in
binaries built with CUDA or HIP support; the device options `host` and `sequential` are always available.

In addition, the benchmark accepts the common options of the `TNL::Benchmarks::Benchmark` framework (such as
`--loops`, `--log-file`, or `--verbose`) and the device setup options of `TNL::Devices::Host` and `TNL::Devices::GPU`.
See the `--help` output for the full list.

With `--write-data true`, the initial condition and the final state are written to Gnuplot files named
`initial-<implementation>-<x-size>-<y-size>.gplt` and `final-<implementation>-<x-size>-<y-size>.gplt`.

## Processing results

The benchmark writes its results to a JSON-lines log file (by default `tnl-benchmark-heat-equation.log`). The log can
be processed into HTML tables with the post-processing script:

```bash
python3 src/Benchmarks/HeatEquation/process-tnl-benchmark-heat-equation.py tnl-benchmark-heat-equation.log
```

The script accepts one or more input log files as positional arguments and writes to an output directory set by
`-o`/`--output-dir` (default: `heat-equation-plots`). It produces:

1. `tnl-benchmark-heat-equation-raw.html` — a table with all records from the input logs.
2. `tnl-benchmark-heat-equation-{float,double}.html` — a table per precision, with the measured times grouped by
   implementation and device, including the speed-up of each implementation over `parallel-for` and the speed-up of
   the CUDA device over the host device.

The script requires `pandas` and the TNL Python utilities from `src/Python` (providing the `TNL` module with the
`BenchmarkLogs` log parser).
