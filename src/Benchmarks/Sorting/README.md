# Sorting Benchmark

Performance benchmark for comparing different sorting algorithms on CPU and GPU.

## Supported algorithms

### Host

- **STL sort** – C++ standard library `std::sort`

### CUDA (GPU)

- **BitonicSort** – TNL bitonic sort implementation
- **Quicksort** – experimental TNL quicksort implementation
- **CUBMergeSort** – `cub::DeviceMergeSort::SortKeys`
- **CUBRadixSort** – `cub::DeviceRadixSort::SortKeys`
- **CedermanQuicksort** – Quicksort implementation based on Cederman & Tsigas
- **MancaQuicksort** – Quicksort by Manca et al.
- **NvidiaBitonicSort** – Bitonic sort NVIDIA CUDA samples

#### Algorithm references

- **CedermanQuicksort**: D. Cederman, P. Tsigas. "GPU-Quicksort: A practical quicksort algorithm for graphics processors." *Journal of Experimental Algorithmics (JEA)*, 2010.
  DOI: [10.1145/1498698.1564500](https://doi.org/10.1145/1498698.1564500)

- **MancaQuicksort**: E. Manca, A. Manconi, A. Orro, G. Armano, L. Milanesi. "CUDA-quicksort: an improved GPU-based implementation of quicksort." *Concurrency and Computation: Practice and Experience*, 2015.
  DOI: [10.1002/cpe.3611](https://doi.org/10.1002/cpe.3611), [SourceForge](https://sourceforge.net/p/cuda-quicksort/)

- **NvidiaBitonicSort**: From [NVIDIA CUDA samples](https://github.com/NVIDIA/cuda-samples), `Samples/2_Concepts_and_Techniques/sortingNetworks/bitonicSort.cu`

#### Known limitations

- **NvidiaBitonicSort** supports only `unsigned int` values and power-of-2 array sizes >= 1024.

## Distributions

The benchmark tests each algorithm across different data distributions:

| Distribution | Description |
|--------------|-------------|
| `random` | Random data (uniform distribution) |
| `shuffle` | Sorted array shuffled |
| `sorted` | Already sorted array |
| `almost-sorted` | Mostly sorted with a few swaps |
| `decreasing` | Reverse sorted |
| `gaussian` | Random data (Gaussian distribution) |
| `bucket` | Values grouped into buckets |
| `staggered` | Staggered distribution |
| `zero-entropy` | All same values |

## Usage

### Compilation

First you need to compile either a CPU-only or CUDA-enabled executable of the benchmark:

```bash
# Build CPU-only benchmark
just build tnl-benchmark-sort

# Build CUDA-enabled benchmark
just build tnl-benchmark-sort-cuda
```

#### Reference algorithm from CUDA samples

The benchmark can include an additional reference algorithm: **NvidiaBitonicSort**, a bitonic sort from NVIDIA's CUDA samples repository.
When building with CUDA support and the `TNL_FETCH_CUDA_SAMPLES` option enabled, the CUDA samples are automatically downloaded via CMake's FetchContent module.

### Running

You can execute the benchmark directly with custom parameters:

```bash
# Run with default parameters (1M elements, host device, int type)
./build/bin/tnl-benchmark-sort-cuda --size 1000000 --device cuda --value-type int

# Run with all available algorithms
./build/bin/tnl-benchmark-sort-cuda --device all --value-type all
```

Both CPU-only and CUDA executables support the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--log-file <file>` | Log file name | `tnl-benchmark-sort.log` |
| `--output-mode <mode>` | File mode (`overwrite` or `append`) | `overwrite` |
| `--loops <n>` | Number of benchmark repetitions | `10` |
| `--verbose <n>` | Verbose mode (0-1) | `1` |
| `--size <n>` | Array size | `1048576` (2^20) |
| `--device <device>` | Device to use (`host`, `cuda`, `all`) | `host` |
| `--value-type <type>` | Value type (`int`, `uint`, `double`, `all`) | `int` |

Alternatively, you can execute the `run-tnl-benchmark-sort` script which automates running benchmarks across multiple array sizes, value types, and devices:

```bash
# Run with default solver (tnl-benchmark-sort-cuda)
./run-tnl-benchmark-sort

# Specify custom solver
./run-tnl-benchmark-sort ./build/bin/tnl-benchmark-sort-cuda
```

### Visualizing results


The benchmark outputs timing measurements in the [JSONL](https://jsonltools.com/what-is-jsonl) format with the following keys:

- distribution, value_type, device, performer (algorithm), time, speedup, bandwidth, cycles/op, cycles, time_stddev, loops, ops_per_loop

Results are logged to the specified log file and metadata is written to `<log-file>.metadata.json`.

Use the `plot-results.py` script to generate performance plots:

```bash
# Generate all plots in current directory
./plot-results.py tnl-benchmark-sort.log

# Specify output directory
./plot-results.py tnl-benchmark-sort.log --output-dir ./plots
```

The script generates PDF plots showing time vs. array size for each algorithm and distribution.
Different line styles represent different value types.
