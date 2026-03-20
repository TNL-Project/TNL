# Sorting Benchmark

Performance benchmark for comparing performance of different sorting algorithms on CPU and GPU.

## Usage

### Host (CPU) Build

```bash
# Build with default options (CPU only)
just build tnl-benchmark-sort

# Run with default parameters (1M elements, host device, int type)
./build/bin/tnl-benchmark-sort

# Run with custom parameters
./build/bin/tnl-benchmark-sort --size 1000000 --device host --value-type int
```

### GPU (CUDA) Build

```bash
# Build with CUDA support
just configure -DTNL_USE_CUDA=ON
just build tnl-benchmark-sort-cuda

# Run on GPU
./build/bin/tnl-benchmark-sort-cuda --size 1000000 --device cuda --value-type int

# Run with all available algorithms
./build/bin/tnl-benchmark-sort-cuda --device all --value-type all
```

### Building with reference algorithms

Some of the reference CUDA algorithms (MancaQuicksort, NvidiaBitonicSort) require the CUDA Samples package:

1. Install CUDA samples (typically in `/usr/local/cuda/samples` on Arch Linux)
2. Set `CUDA_SAMPLES_DIR` when configuring CMake:

   ```bash
   cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCUDA_SAMPLES_DIR=/usr/local/cuda/samples
   ```

3. Build:

   ```bash
   just build tnl-benchmark-sort-cuda
   ```

Note: Some reference algorithms may have compatibility issues with newer CUDA versions or compiler toolchains.

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--log-file <file>` | Log file name | `tnl-benchmark-sort.log` |
| `--output-mode <mode>` | File mode (`overwrite` or `append`) | `overwrite` |
| `--loops <n>` | Number of benchmark repetitions | `10` |
| `--verbose <n>` | Verbose mode (0-1) | `1` |
| `--size <n>` | Array size | `1048576` (2^20) |
| `--device <device>` | Device to use (`host`, `cuda`, `all`) | `host` |
| `--value-type <type>` | Value type (`int`, `double`, `all`) | `int` |

## Supported algorithms

### Host

- **STL sort** - C++ standard library sort

### CUDA (GPU)

- **Bitonic sort** - TNL BitonicSort
- **Quicksort** - TNL experimental::Quicksort
- **Cederman quicksort** - CUDA reference implementation
- **Thrust radix sort** - NVIDIA Thrust library
- **Manca quicksort** - NVIDIA reference implementation (requires `CUDA_SAMPLES_DIR`)
- **Nvidia bitonic sort** - NVIDIA reference implementation (requires `CUDA_SAMPLES_DIR`)

## Distributions

The benchmark tests each algorithm across different data distributions:

- `random` - Random data (uniform distribution)
- `shuffle` - Sorted array shuffled
- `sorted` - Already sorted array
- `almost-sorted` - Mostly sorted with a few swaps
- `decreasing` - Reverse sorted
- `gaussian` - Random data (Gaussian distribution)
- `bucket` - Values grouped into buckets
- `staggered` - Staggered distribution
- `zero-entropy` - All same values

## Output

The benchmark outputs timing measurements in CSV-like format with columns:

- distribution, precision (value type), device, performer (algorithm), time, speedup, bandwidth, cycles/op, cycles, time_stddev, loops, ops_per_loop

Results are logged to the specified log file and metadata is written to `<log-file>.metadata.json`.
