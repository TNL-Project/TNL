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

## Binaries

The directory contains the benchmark sources for CPU and CUDA variants (e.g., `tnl-benchmark-graphs-bfs.cpp`, `tnl-benchmark-graphs-sssp.cu`, `tnl-benchmark-graphs-mst.cpp`). Build with your chosen preset/target (e.g., `ninja -C build/debug tnl-benchmark-graphs-bfs`) to produce the executables consumed by the scripts above.
