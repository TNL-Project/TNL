# Graph Benchmarks

This directory contains the executables and helper scripts used to benchmark graph algorithms (BFS, SSSP, MST, etc.) on real-world and synthetic datasets.

## Data preparation

`extract-graphs` copies a local mirror of the SuiteSparse Matrix Collection into a working `graphs/` directory while preserving subfolders. It does **not** download files; you must already have the source tree with the same relative paths as listed in the script.

```bash
# Example: stage graphs from a local mirror into ./graphs
./extract-graphs /path/to/suitesparse-mirror ./graphs
```

The matrices from the SuiteSparse Matrix Collection can be downloaded using the script `script/get-matrices` in [tnl-benchmark-spmv](https://gitlab.com/tnl-project/tnl-benchmark-spmv).

## Running benchmarks on existing graphs

`run-tnl-benchmark-graphs` executes a chosen benchmark binary over all `.mtx` files under a given directory, builds an `input-files` list (respecting an optional size limit), and appends results to a log.

Common flags:

```bash
./run-tnl-benchmark-graphs \
  --benchmark /path/to/tnl-benchmark-graphs      \  # CPU binary
  --log-file ./log-files/graphs-benchmark.log    \  # output log
  --input-dir ./graphs                           \  # root with .mtx files
  --size-limit 1073741824                        \  # bytes; -1 disables filter
  --openmp-enabled yes                           \  # enable/disable OpenMP
  --openmp-max-threads auto                      \  # core count when auto
  --debug no                                     \  # run under gdb when yes
  --continue no                                  \  # reuse existing log/input-files
  --precision double                                # float|double
```

Notes:
- A segfault is logged to `segfaults.log` and execution continues.
- When `--continue no` (default), existing `input-files` or log files are removed before starting.

## Convenience wrapper

`run-all-benchmarks` is a thin wrapper that calls `run-tnl-benchmark-graphs` for multiple binaries (CPU and CUDA). It assumes the benchmarks live in `$HOME/.local/bin` by default; adjust `BIN_DIR` inside the script or export it before running.

```bash
# Run both CPU and CUDA variants on ./graphs
BIN_DIR=$HOME/.local/bin ./run-all-benchmarks
```

## Random graph generation

`run-tnl-benchmark-random-graphs` generates synthetic graphs via `tnl-graph-generator.py`, runs a benchmark binary on each, and can render Graphviz outputs.

Common flags:

```bash
# Generate and benchmark 5 graphs, starting at 100 nodes / 250 edges, saving into ./out
./run-tnl-benchmark-random-graphs \
  --num-graphs 5 \
  --start-nodes 100 --nodes-increment 50 \
  --start-edges 250 --edges-increment 125 \
  --weights normal \
  --base-name rnd \
  --output-dir ./out \
  --generator tnl-graph-generator.py \
  --benchmark tnl-benchmark-graphs-dbg \
  --loops 0 \
  --graphviz yes
```

Useful flags: `--num-graphs`, `--start-nodes`, `--nodes-increment`, `--start-edges`, `--edges-increment`, `--weights` (edge weight distribution), `--base-name`, `--output-dir`, `--generator` (script path), `--benchmark` (binary), `--loops` (benchmark loops), `--graphviz yes|no`. Use `-h` for help.

## Processing the results
`tnl-benchmark-graphs-process-results.py` reads one or more JSON-lines log files from `tnl-benchmark-graphs`, builds a structured Pandas table, computes speedups, and generates HTML summaries and PDF plots.

### Quick use

```bash
python3 tnl-benchmark-graphs-process-results.py \
  -i graphs-benchmark.log other.log   # one or more input logs
```

Outputs are written to the current directory:
- `graphs-benchmark-input.html`: concatenated raw input logs
- `graphs-benchmark.html`: reshaped table with computed speedups
- `Time/<problem>/<device>/<kernel>-<launch>.pdf`: time profiles (sorted by time)
- `Speedup/<problem>/<device>/<kernel>-<launch>-vs-{Boost|Gunrock}.pdf` (+ `-log`): speedup profiles
- `Speedup/<problem>/<device>/<kernel>-<launch>.html`: snapshot of the filtered table

### Assumptions and inputs
- Speedups are computed vs Boost on CPU, and vs Gunrock on CUDA when available.
- Requires Python with pandas, numpy, matplotlib, and the TNL Python helpers (`TNL.BenchmarkLogs`, `TNL.MultiindexCreator`).

## Speedup summary table

`tnl-benchmark-graphs-speedup-table.py` reads one or more JSON-lines log files and
builds a single per-graph table (one row per graph) with a structured (MultiIndex)
header. For every TNL run on a GPU it computes the speedup relative to TNL on CPU
(the "host" run with plain CSR segments), to Boost, to each individual Gunrock
launch configuration, and to the best (fastest) of the Gunrock launch
configurations; it also computes the speedup of TNL on CPU relative to Boost.
Semiring problem variants (e.g. `Semiring BFS dir`) are compared against the
baselines of the corresponding plain problem, since Boost/Gunrock have no semiring
implementations. It uses `TNL.BenchmarkLogs` and `TNL.MultiindexCreator`
(`src/Python/BenchmarkLogs.py` and `src/Python/MultiindexCreator.py`).

For each problem, it also renders two performance-profile plots. In both, the
x-axis is the speedup relative to the best Gunrock launch configuration (log
scale) and the y-axis is the percentage of graphs that reach that speedup or
better:
- `<problem>-vs-gunrock-best.pdf`: one curve per TNL kernel (best time across its
  launch configurations) plus a "best TNL (GPU)" curve (the best time across
  *all* GPU kernels, host excluded).
- `<problem>-vs-gunrock-best-CSR.pdf`: one curve per launch configuration of the
  CSR kernel only (Ellpack-family kernels are ignored), plus a "best TNL (GPU,
  CSR)" curve — this compares traversal/launch strategies rather than segment
  storage formats.

```bash
python3 tnl-benchmark-graphs-speedup-table.py \
  -i graphs-benchmark.log other.log \
  -o graphs-benchmark-speedup \  # base name for the .html/.csv outputs
  --plot-dir Plots \             # directory for the profile plots
  --xlim-min 0.1 --xlim-max 4    # range of the logarithmic speedup axis
```

Outputs:
- `<output>.html` / `<output>.csv`: the speedup table
- `Plots/<problem>-vs-gunrock-best.pdf` and `Plots/<problem>-vs-gunrock-best-CSR.pdf`:
  the two profile plots per problem

