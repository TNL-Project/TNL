# NDArray Benchmarks

This directory contains benchmarks for `TNL::Containers::NDArray`, TNL's n-dimensional array
container.
There are four benchmarks:

- **tnl-benchmark-ndarray** measures the memory bandwidth of element-wise access and traversal
  (a plain copy `a = b` via `nd_map`) over NDArrays of dimension 1 to 6, in the natural index
  order as well as in permuted orders, with a plain 1D `Array` copy as the baseline.
- **tnl-benchmark-ndarray-boundary** measures the bandwidth of the boundary and interior traversal
  (`forBoundary` and `forInterior`) of NDArrays of dimension 1 to 3, including permuted variants.
- **tnl-benchmark-ndarray-reduction** measures the bandwidth of reductions (`nd_reduce`) over
  NDArrays of dimension 1 to 6, along every reduction axis and for several index permutations.
- **tnl-benchmark-reduction** measures reductions over plain 1D arrays and manually indexed 2D and
  3D data (`reduce`, `Reduction2D`, `Reduction3D`), providing a baseline for comparison with
  `nd_reduce` in the NDArray reduction benchmark.

## Usage

### Compilation

The CMake build produces up to three variants of each benchmark: a host-only executable, a CUDA
executable with the `-cuda` suffix, and a HIP executable with the `-hip` suffix. The base names are:

- `tnl-benchmark-ndarray`
- `tnl-benchmark-ndarray-boundary`
- `tnl-benchmark-ndarray-reduction`
- `tnl-benchmark-reduction`

The host-only targets are built when `TNL_BUILD_CPP_TARGETS` is enabled (skipped in CUDA- and
HIP-enabled CI jobs), the CUDA variants when `TNL_BUILD_CUDA` is enabled, and the HIP variants when
`TNL_BUILD_HIP` is enabled.

```bash
# Build host-only benchmarks
just build tnl-benchmark-ndarray

# Build CUDA-enabled benchmark
just build tnl-benchmark-ndarray-reduction-cuda
```

### Running

You can execute the benchmarks directly with custom parameters:

```bash
# Run the NDArray traversal benchmark on the host only
./build/bin/tnl-benchmark-ndarray --device host

# Run the NDArray reduction benchmark on a CUDA device with a custom log file
./build/bin/tnl-benchmark-ndarray-reduction-cuda --device cuda --log-file ndarray-reduction.log
```

The benchmarks write timing measurements to the terminal and, when `--log-file` is given, to a
[JSONL](https://jsonltools.com/what-is-jsonl) log file (with global hardware metadata written to
`<log-file>.metadata.json`). The NDArray reduction benchmark logs the reduction axis, the index
permutation, and the array sizes as metadata columns (axis, permutation, size, m, n, o, p, q), which
is used by the post-processing script below.

## Command-line options

All four benchmarks share the following options. Only `tnl-benchmark-ndarray-reduction` accepts
additional options (see below); the array sizes of the other three benchmarks are fixed at compile
time:

| Option | Description | Default |
|--------|-------------|---------|
| `--device <device>` | Device to run the benchmarks on (`sequential`, `host`, `cuda`, `hip`, `all`) | `all` |
| `--log-file <file>` | Log file name for JSONL output | `<program>.log` |
| `--output-mode <mode>` | Mode for opening the log file (`overwrite` or `append`) | `overwrite` |
| `--loops <n>` | Number of iterations for every computation | `10` |
| `--min-time <t>` | Minimal real time in seconds for every computation | `0.0` |
| `--warmup-loops <n>` | Number of warmup iterations before timing (0 to disable) | `1` |
| `--warmup-min-time <t>` | Minimal real time in seconds for warmup (0 to disable) | `0.0` |
| `--verbose <n>` | Verbosity of the terminal output | `1` |
| `--catch-exceptions <bool>` | Catch exceptions during timing | `true` |
| `--openmp-enabled <bool>` | Enable OpenMP support (host device) | `true` |
| `--openmp-max-threads <n>` | Maximum number of OpenMP threads (host device) | `omp_get_max_threads()` |
| `--cuda-device <n>` / `--hip-device <n>` | Device to run the computation on (GPU builds) | `0` |

The `tnl-benchmark-ndarray-reduction` benchmark additionally accepts options controlling the array
sizes (size lists are comma-separated lists of positive integers):

| Option | Description | Default |
|--------|-------------|---------|
| `--sizes-1d <list>` | Sizes for 1D reduction | `5000000,50000000,500000000` |
| `--sizes-2d-3d <list>` | Sizes for 2D and 3D reduction | `64,256,1024,4096,16384` |
| `--sizes-4d <list>` | Sizes for 4D reduction | `4,16,128,256` |
| `--sizes-5d-6d <list>` | Sizes for 5D and 6D reduction | `2,16,128` |
| `--min-elements <n>` | Skip combinations with fewer total elements (applied only to 4D, 5D, and 6D) | `50000000` |
| `--max-elements <n>` | Skip combinations with more total elements | `2000000000` |

For 2D and 3D arrays, only combinations with at most `--max-elements` total elements are run.
For 4D to 6D arrays, combinations are additionally skipped below `--min-elements`, since the full
Cartesian product of the size lists would produce too many tiny cases.

## Processing the results

The script requires Python with `pandas` and `matplotlib` installed, and the TNL Python helpers
(`TNL.BenchmarkLogs`) available on the `PYTHONPATH` (see `src/Python/`). It works with JSONL logs
from the NDArray reduction benchmark, which logs the axis and permutation metadata.

### Visualizing results

`plot-results.py` reads one or more JSONL log files and renders a bandwidth heatmap per array
dimension and device, with the reduction axis on the vertical axis and the index permutation on the
horizontal axis:

```bash
# Generate heatmaps in the current directory
./plot-results.py ndarray-reduction.log

# Specify output directory
./plot-results.py ndarray-reduction.log --output-dir ./plots
```

The script writes `heatmap-<dim>D[-<device>].svg` files (e.g. `heatmap-3D-cuda.svg`) into the
output directory (default: the current directory).
