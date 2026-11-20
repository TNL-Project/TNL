# Sorting Benchmark

Performance benchmark for comparing different sorting algorithms on CPU and GPU,
and a benchmark of the three-way `pivotPartition` operation (see
[Pivot partition benchmark](#pivot-partition-benchmark)).

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

Both CPU-only and CUDA executables accept the common benchmark options (output, number of iterations and device
settings), which are described in the [benchmarks README](../README.md#common-command-line-options),
and the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--size <n>` | Array size | `1048576` (2^20) |
| `--device <device>` | Device to use (`host`, `cuda`, `all`) | `host` |
| `--value-type <type>` | Value type (`int`, `uint`, `float`, `double`, `all`) | `int` |

Alternatively, you can execute the `run-tnl-benchmark-sort` script which automates running benchmarks across multiple array sizes, value types, and devices:

```bash
# Run with default solver (tnl-benchmark-sort-cuda)
./run-tnl-benchmark-sort

# Specify custom solver
./run-tnl-benchmark-sort ./build/bin/tnl-benchmark-sort-cuda
```

### Visualizing results


The benchmark outputs timing measurements in the [JSONL](https://jsonltools.com/what-is-jsonl) format with the following keys:

- distribution, value_type, device, performer (algorithm), time, bandwidth, cycles/op, cycles, time_stddev, loops, ops_per_loop

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

## Pivot partition benchmark

The `tnl-benchmark-pivot-partition` executable benchmarks the
`TNL::Algorithms::pivotPartition` operation, which splits an array into elements
less than, equivalent to, and greater than a pivot value. No full sort is
measured - each timed invocation performs only the partitioning.

### Performers

Host:

- **std::partition x2** – two applications of `std::partition`: the first one on
  `compare(value, pivot)`, the second one on the remaining range with
  `!compare(pivot, value)`
- **TNL in-place** – `pivotPartition(data, pivot)`
- **TNL out-of-place** – `pivotPartition(input, output, pivot)` with the output
  array preallocated outside of the timed operation

CUDA (GPU):

- **TNL in-place** – `pivotPartition(data, pivot)`, including the internal
  scratch allocation (the scratch array of `N` elements is part of the real
  cost of the in-place GPU overload)
- **TNL out-of-place** – `pivotPartition(input, output, pivot)` with the output
  array preallocated outside of the timed operation

Every performer partitions the same input with the same pivot (the middle
element of the input). The input is reset before each timed invocation, outside
of the timed section.

The result of each performer is verified after the timing, outside of the timed
lambda: every output element must satisfy the predicate of its bucket and the
output must be a permutation of the input (compared via sorted host copies).
The order of the elements within the buckets is not checked, since the
partitioning is not stable.

### Distributions

The benchmark uses a subset of the distributions listed above: `random`,
`sorted`, `decreasing`, `bucket` and `zero-entropy`.

### Compilation

```bash
# Build CPU-only benchmark
just build tnl-benchmark-pivot-partition

# Build CUDA-enabled benchmark
just build tnl-benchmark-pivot-partition-cuda
```

### Running

```bash
# Run with default parameters (1M elements, int type) on the host
./build/bin/tnl-benchmark-pivot-partition --device host

# Run on the GPU with a custom size
./build/bin/tnl-benchmark-pivot-partition-cuda --device cuda --size 1048576
```

The benchmark supports the same general options as the sorting benchmark
(`--log-file`, `--output-mode`, `--loops`, `--verbose`) and the following
benchmark-specific options:

| Option | Description | Default |
|--------|-------------|---------|
| `--size <n>` | Array size | `1048576` (2^20) |
| `--device <device>` | Device to use (`host`, `cuda`, `hip`, `all`) | `all` |
| `--value-type <type>` | Value type (`int`, `uint`, `double`, `all`) | `int` |

The benchmark outputs timing measurements in the same JSONL format as the
sorting benchmark, with the same metadata keys (distribution, value type,
device, performer, time, ...).
