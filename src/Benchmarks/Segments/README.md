# Segments Benchmark

Performance benchmark comparing segmented data structures from `TNL::Algorithms::Segments` on CPU and GPU.
It measures the speed of common segment operations, such as element traversal and per-segment reduction,
for several segments formats and multiple kernel launch configurations.

## Segments formats

### Host and sequential

- **CSR** (with the `CSRScalarKernel` reduction kernel)

### CUDA and HIP (GPU)

- **CSR** (with the `CSRScalarKernel` reduction kernel)
- **Ellpack** (with the `EllpackKernel` reduction kernel)
- **SlicedEllpack** (with the `SlicedEllpackKernel` reduction kernel)
- **BiEllpack** (with the `BiEllpackKernel` reduction kernel)
- **ChunkedEllpack** (with the `ChunkedEllpackKernel` reduction kernel)

## Measured operations

Each format is benchmarked with the following operations:

### Traversal

- `forElements` – traversal of all elements in all segments
- `forElements` with explicit segment indexes picked with stride 2, 4, and 8
- `forElementsIf` – traversal restricted by a condition on the segment index, with strides 2, 4, and 8
- `forSelectedElements` – traversal of segments selected by a condition, with strides 2, 4, and 8

### Reduction

- `reduceSegments` – sum reduction over all segments
- `reduceSegments` with explicit segment indexes picked with stride 2, 4, and 8
- `reduceSegmentIf` – conditional sum reduction, with strides 2, 4, and 8

Each operation is timed with all kernel launch configurations supported by the given format. These are
referred to as threads mappings, for example fixed numbers of threads per segment (1 to 256 TPS),
block-merged mappings, or dynamic grouping. Reduction results are verified against the expected segment sizes.

## Segments setups

The benchmark generates segment sizes in three setups:

| Setup | Description |
|-------|-------------|
| `constant` | All segments have the same size |
| `linear` | Segment sizes grow linearly with the segment index (`i % max_segment_size + 1`) |
| `quadratic` | Segment sizes follow a quadratic pattern (`i*i % max_segment_size + 1`) |

For each setup, the benchmark sweeps the number of segments and the maximum segment size in powers of two
between the `--min-segments-count`/`--max-segments-count` and `--min-segment-size`/`--max-segment-size` limits.

## Usage

### Compilation

First you need to compile a CPU-only, CUDA-enabled, or HIP-enabled executable of the benchmark:

```bash
# Build CPU-only benchmark
just build tnl-benchmark-segments

# Build CUDA-enabled benchmark
just build tnl-benchmark-segments-cuda

# Build HIP-enabled benchmark
just build tnl-benchmark-segments-hip
```

### Running

You can execute the benchmark directly with custom parameters:

```bash
# Run with default parameters
./build/bin/tnl-benchmark-segments

# Run all setups on the CUDA device
./build/bin/tnl-benchmark-segments-cuda --device cuda

# Run only the constant setup with bounded problem sizes
./build/bin/tnl-benchmark-segments-cuda --device cuda --segments-setup constant --max-segments-count 16384 --max-segment-size 32
```

All executables support the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--log-file <file>` | Log file name for JSONL output | `<program>.log` |
| `--output-mode <mode>` | Mode for opening the log file (`overwrite` or `append`) | `overwrite` |
| `--loops <n>` | Number of iterations for every computation | `10` |
| `--warmup-loops <n>` | Number of warmup iterations before timing (0 to disable) | `1` |
| `--verbose <n>` | Verbosity of the terminal output, the higher the more verbose | `1` |
| `--segments-setup <setup>` | Segments setup (`all`, `constant`, `linear`, `quadratic`) | `all` |
| `--min-segment-size <n>` | Minimum segment size | `1` |
| `--max-segment-size <n>` | Maximum segment size | `128` |
| `--min-segments-count <n>` | Minimum number of segments | `256` |
| `--max-segments-count <n>` | Maximum number of segments | `1048576` |
| `--device <device>` | Device the computation will run on (`host`, `sequential`, `cuda`, `hip`, `all`) | `all` |

Options for OpenMP (`--openmp-enabled`, `--openmp-max-threads`) and for GPU device selection (`--cuda-device`,
`--hip-device`) are available as well, see the `--help` output for details.

### Visualizing results

The benchmark outputs timing measurements in the [JSONL](https://jsonltools.com/what-is-jsonl) format. Each record
includes the metadata columns `segments setup`, `segments count`, `max segment size`, `elements count`,
`segments type`, `function`, and `threads mapping`, together with the measured `time`, `time_stddev`, and `bandwidth`.
Metadata is written to `<log-file>.metadata.json`.

Use the `plot-results.py` script to process the logs. It requires Python with `pandas` and `matplotlib`
installed, and the `TNL` Python module from `src/Python`:

```bash
# Generate tables and plots in the default output directory
./plot-results.py tnl-benchmark-segments-cuda.log

# Specify output directory
./plot-results.py tnl-benchmark-segments-cuda.log --output-dir ./segments-plots
```

The script accepts one or more input log files and writes an HTML table with the raw results, structured HTML tables
per setup and per function, and SVG plots of bandwidth versus segments count for each combination of measured function
and segments format on CUDA. Each plot has one facet per maximum segment size, with a separate line for each threads
mapping, and all plots share a common bandwidth axis. Speedups of the CUDA results with respect to the sequential CSR
baseline are computed as well.
