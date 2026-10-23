# Scan Benchmark

This benchmark measures the scan (prefix sum) in TNL and compares it with the implementations
from other libraries. On the GPU, it also allows measuring the individual TNL scan kernels with
an explicitly selected number of values per thread, which is useful for tuning the defaults of
`GPUScanKernelLauncher` (see `src/TNL/Algorithms/detail/GPUScan.h`) on different GPUs.

## What is measured

For each device, value type, scan type (inclusive or exclusive), and array size, the scan of an
array of ones is measured in these variants:

| Variant | Operation | Description |
|---------|-----------|-------------|
| `inplace` | `inclusive scan (inplace)` | The scan of an array in place |
| `1-vector` | `inclusive scan (1 vector)` | The scan of one array to another array |
| `2-vectors` | `inclusive scan (2 vectors)` | The scan of the expression `a + b` to another array (TNL only) |
| `3-vectors` | `inclusive scan (3 vectors)` | The scan of the expression `a + b + c` to another array (TNL only) |

The following implementations (performers) are measured:

| Performer | Description | Devices | `values per thread` |
|-----------|-------------|---------|---------------------|
| `TNL` | The public API (`Algorithms::inplaceInclusiveScan`, `inclusiveScan`, etc.) | all | `default` |
| `TNL lookback` | The single-pass kernel with decoupled lookback (`GPUScanKernelLookback`) | GPU, `inplace` only | number |
| `TNL scan-then-propagate` | The two-phase scan with `ScanPhaseType::WriteInFirstPhase` (`GPUScanKernelIndependentTiles` and `GPUScanKernelUniformShift`) | GPU, `inplace` only | number |
| `TNL reduce-then-scan` | The two-phase scan with `ScanPhaseType::WriteInSecondPhase` (`GPUScanKernelUpsweep` and `GPUScanKernelDownsweep`) | GPU, `inplace` only | number |
| `std::partial_sum` | C++ standard library, inclusive scan only | sequential | `-` |
| `std::inclusive_scan`, `std::exclusive_scan` | C++ standard library with a parallel execution policy when TBB is available (reported on the `host` device, otherwise on the `sequential` device) | sequential or host | `-` |
| `thrust::inclusive_scan`, `thrust::exclusive_scan` | Thrust (rocThrust in the HIP build) with `thrust::host` and `thrust::device`, when available at build time | sequential or host, GPU | `-` |
| `CUB` | `cub::DeviceScan::InclusiveSum` or `ExclusiveSum` | CUDA | `-` |
| `hipCUB` | `hipcub::DeviceScan::InclusiveSum` or `ExclusiveSum` | HIP | `-` |
| `rocPRIM` | `rocprim::inclusive_scan` or `exclusive_scan` with the default configuration | HIP | `-` |

The other libraries are measured only for the arithmetic value types and only in the `inplace`
and `1-vector` variants. The Thrust host algorithms are reported on the `host` device when the
Thrust host backend is parallel (TBB or OpenMP), otherwise on the `sequential` device.

The explicitly selected GPU kernels use the default block size of the launcher, which is
`256 * 4 / sizeof(value type)` (i.e. 1 kB of values per value of each thread), and the given
number of values per thread. The number of values per thread is a template parameter, so the
benchmark contains a fixed set of values: 7, 9, 11, 13, 15, 17, 19 (see `ValuesPerThreadList` in
`scan-common.h`; the values must be odd to avoid shared memory bank conflicts). The two-phase
kernels are measured even for the sizes where the launcher would select the lookback kernel.

The result of each measurement is compared with a reference computed sequentially on the host,
and a mismatch is reported on the standard error output. For floating-point types, the
comparison is skipped for sizes where the partial sums are not exact.

## Executables

| Executable | Description |
|------------|-------------|
| `tnl-benchmark-scan` | Host (CPU) build |
| `tnl-benchmark-scan-cuda` | CUDA build, compared with CUB and Thrust |
| `tnl-benchmark-scan-hip` | HIP build, compared with hipCUB, rocPRIM, and rocThrust |

The CUDA and HIP executables run both the host and the GPU implementations.

## Command-line options

The benchmark-specific options are:

| Option | Description | Default |
|--------|-------------|---------|
| `--device <device>` | `sequential`, `host`, `cuda` (CUDA build), `hip` (HIP build), or `all` | `all` |
| `--value-type <type>` | `char`, `short`, `int`, `long`, `float`, `double`, `double2` (`StaticVector<2, double>`), or `all` | `all` |
| `--scan-type <type>` | `inclusive`, `exclusive`, or `all` | `all` |
| `--variants <list>` | `inplace`, `1-vector`, `2-vectors`, `3-vectors`, or `all` | `all` |
| `--libraries <list>` | `tnl`, `std`, `thrust`, `cub` (CUB or hipCUB), `rocprim`, or `all` | `all` |
| `--gpu-kernels <list>` | `default` (the public API), `lookback`, `scan-then-propagate`, `reduce-then-scan`, or `all` | `default` |
| `--values-per-thread <list>` | Values per thread of the explicitly selected GPU kernels | `7 9 11 13 15 17 19` |
| `--min-size <n>` | Minimum size of the scanned arrays | `100000` |
| `--max-size <n>` | Maximum size of the scanned arrays | `25600000` |
| `--size-step-factor <n>` | Each following size is `size-step-factor * previous size` | `4` |

The values of `--values-per-thread` must be among the compiled values listed above. In addition,
the common benchmark settings (e.g. `--loops`, `--log-file`, `--verbose`) and the device settings
(`--openmp-enabled`, `--openmp-max-threads` and `--cuda-device` or `--hip-device`) are
available. The number of threads of the host scan is set by `--openmp-max-threads`.

## Usage

```bash
just build tnl-benchmark-scan-cuda

# comparison with the other libraries on all devices
./build/bin/tnl-benchmark-scan-cuda --log-file scan-$(hostname).log

# tuning of the GPU kernels: only the in-place scan on the GPU, all kernels and values per thread
./build/bin/tnl-benchmark-scan-cuda --device cuda --variants inplace --gpu-kernels all \
    --log-file scan-tuning-$(hostname).log

# only the lookback kernel with selected values per thread for double
./build/bin/tnl-benchmark-scan-cuda --device cuda --value-type double --scan-type inclusive \
    --variants inplace --libraries tnl --gpu-kernels lookback --values-per-thread 11 13 15
```

The benchmark writes the measurements in the JSONL format to the log file (`<program>.log` by
default), together with the metadata of the devices in `<program>.metadata.json`, so the results
from different GPUs can be compared.

## Processing the results

The script `tnl-benchmark-scan-html-table-generator.py` (requires `pandas` and the TNL Python
module `TNL.BenchmarkLogs`) generates HTML tables from one or more logs:

```bash
src/Benchmarks/Algorithms/tnl-benchmark-scan-html-table-generator.py scan-$(hostname).log
```

For each operation, it produces the tables of times and bandwidths of all devices and performers,
the speedups of the GPU against the CPU, and the speedups of TNL against the other libraries.
When the log contains the explicitly selected GPU kernels, it also produces the tuning tables for
each operation and GPU: the times of all kernels and values per thread, the best number of values
per thread for each value type and size, and the recommended number of values per thread for each
value type (the one with the smallest geometric mean of the slowdown against the best one over all
sizes). The operations and value types can be selected with `-f` and `-t`, see `--help`.
