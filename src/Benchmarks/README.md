# TNL Benchmarks

This directory contains the performance benchmarks of TNL.
Each subdirectory documents its benchmark in its own README file:

- [BLAS](BLAS/README.md)
- [Dense linear solvers](DenseLinearSolvers/README.md)
- [Dense matrices](DenseMatrices/README.md)
- [Eigen solvers](Eigen/README.md)
- [Graphs](Graphs/README.md)
- [Grid](Grid/README.md)
- [Heat equation](HeatEquation/README.md)
- [Linear solvers](LinearSolvers/README.md)
- [NDArray](NDArray/README.md)
- [ODE solvers](ODESolvers/README.md)
- [Scan](Scan/README.md)
- [Segments](Segments/README.md)
- [Sorting](Sorting/README.md)
- [System](System/README.md)

The benchmarks measure and log the results with the `TNL::Benchmarks::Benchmark` class,
so they share the common command-line options described below.
The options specific to each benchmark are described in its README file.
A complete list of the options of a benchmark can be obtained with `--help`.
The conventions for writing new benchmarks are described in [ARCHITECTURE.md](ARCHITECTURE.md).

## Common command-line options

### Output

| Option | Description | Default |
|--------|-------------|---------|
| `--log-file <file>` | Log file name for JSONL output | `<program>.log` |
| `--output-mode <mode>` | Mode for opening the log file (`overwrite` or `append`) | `overwrite` |
| `--verbose <n>` | Verbosity of the terminal output, the higher the more verbose | `1` |
| `--catch-exceptions <bool>` | Catch exceptions during timing and log them instead of terminating | `true` |

The results are written to the terminal and to the log file, one JSON record per line.
The hardware and system metadata are written to a separate file `<log-file>.metadata.json`
(the extension of the log file is replaced).
The log can be converted to an HTML report with tables and plots by `src/Tools/tnl-benchmark-to-html.py`:

```bash
src/Tools/tnl-benchmark-to-html.py tnl-benchmark-blas-cuda.log
```

### Number of iterations

| Option | Description | Default |
|--------|-------------|---------|
| `--loops <n>` | Maximal number of iterations for every computation | `10` |
| `--max-time <t>` | Real time in seconds after which no further iteration of the computation is started (0 for no limit) | `0` |
| `--warmup-loops <n>` | Maximal number of warmup iterations before timing (0 to disable) | `1` |
| `--warmup-max-time <t>` | Real time in seconds after which no further warmup iteration is started (0 for no limit) | `0` |
| `--reuse-long-warmup <bool>` | Use a warmup iteration taking at least `--max-time` as the only timing sample | `false` |

Every computation is executed in two phases.
First, the warmup iterations stabilize the thermal and frequency states of the hardware
and amortize one-time costs such as the CUDA JIT compilation.
Their times are discarded, unless a warmup iteration is reused as a sample with `--reuse-long-warmup` (see below).
Then, the measured iterations are timed and their mean, median and standard deviation are logged
together with the number of measured iterations (the `loops` column).
Each iteration calls the reset function of the benchmark first, which is not timed.

Both phases end after the given number of iterations,
or earlier when the time limit of the phase is positive and the total time of the phase reaches it.
The time limits never interrupt a running computation:
they are checked after each iteration and only prevent starting the next one.
Hence, the total time of a phase may exceed its limit by up to one iteration,
and a single iteration longer than the limit always runs to its end.
At least one iteration is always measured.
Hence, short computations are measured `--loops` times
and long computations are measured fewer times, at least once.

A long computation is still executed once more during the warmup.
With `--reuse-long-warmup true` and a positive `--max-time`,
a warmup iteration which takes at least `--max-time` by itself
ends the warmup and is used as the only timing sample.
Such a computation is long enough that the effects compensated by the warmup are negligible.
For example, the following options measure each computation 10 times,
but computations taking at least one minute are executed only once:

```bash
--loops 10 --max-time 60 --reuse-long-warmup true
```

The results based on a single sample have no standard deviation (it is logged as `NaN`).

### Device settings

| Option | Description | Default |
|--------|-------------|---------|
| `--openmp-enabled <bool>` | Enable support of OpenMP (host device) | `true` |
| `--openmp-max-threads <n>` | Maximum number of OpenMP threads (host device) | `omp_get_max_threads()` |
| `--cuda-device <n>` | CUDA device to run the computation on (CUDA builds) | `0` |
| `--hip-device <n>` | HIP device to run the computation on (HIP builds) | `0` |

Benchmarks using MPI accept also the options of `TNL::MPI::configSetup`, see their `--help` output.
