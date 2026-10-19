# Scan Tuning Benchmark

This benchmark measures the GPU scan (prefix sum) and allows tuning the configuration of the
single-pass lookback kernel (`GPUScanKernelLookback`, see
`src/TNL/Algorithms/detail/GPUScan/GPUScanKernelLookback.h`) on different GPUs. The kernel is
used by `TNL::Algorithms::inclusiveScan`, `exclusiveScan`, and their in-place variants for
arrays that are larger than one tile and fit into one CUDA grid.

## What is measured

For each value type, scan type (inclusive or exclusive), and array size, the in-place scan of an
array of ones is measured with:

| Performer | Description | `block size` | `values per thread` |
|-----------|-------------|--------------|---------------------|
| `TNL` | The public API (`Algorithms::inplaceInclusiveScan` or `inplaceExclusiveScan`) with the default configuration | `default` | `default` |
| `CUB` | `cub::DeviceScan::InclusiveSum` or `ExclusiveSum` (CUDA builds, arithmetic value types only) | `-` | `-` |
| `TNL lookback` | `GPUScanKernelLauncher::performLookback` with the given configuration | number | number |

The result of each measurement is compared with the result of the public API, and a mismatch
is reported on the standard error output. For floating-point types, the comparison is skipped
for sizes where the partial sums of ones are not exact.

The configurations of the lookback kernel are template parameters, so the benchmark contains a
fixed set of them (see `BlockSizeFactors` and `ValuesPerThreadList` in `scan-tuning.h`):

- block size: 1x and 2x the default block size of the launcher, which is
  `256 * 4 / sizeof(value type)` (block sizes larger than 1024 are skipped),
- values per thread: 7, 9, 11, 13, 15, 17, 19 (odd numbers avoid shared memory bank conflicts).

The current default of the launcher is the default block size and 15 values per thread for
value types up to 8 bytes, 11 for larger types.

## Executables

| Executable | Description |
|------------|-------------|
| `tnl-benchmark-scan-cuda` | CUDA build, compared with CUB |
| `tnl-benchmark-scan-hip` | HIP build |

There is no host-only executable, since the benchmark tunes a GPU kernel.

## Command-line options

The benchmark-specific options are:

| Option | Description | Default |
|--------|-------------|---------|
| `--value-type <type>` | `char`, `short`, `int`, `long`, `float`, `double`, `double2` (`StaticVector<2, double>`), or `all` | `all` |
| `--scan-type <type>` | `inclusive`, `exclusive`, or `all` | `all` |
| `--min-size <n>` | Minimum size of the scanned arrays | `100000` |
| `--max-size <n>` | Maximum size of the scanned arrays | `25600000` |
| `--size-step-factor <n>` | Each following size is `size-step-factor * previous size` | `4` |
| `--block-size-factors <list>` | Block sizes of the lookback kernel as multiples of the default block size | `1 2` |
| `--values-per-thread <list>` | Values per thread of the lookback kernel | `7 9 11 13 15 17 19` |
| `--device <device>` | `cuda`, `hip`, or `all` | `cuda` |

The values of the list options must be among the compiled configurations listed above. In
addition, the common benchmark settings (e.g. `--loops`, `--log-file`, `--verbose`) and the
device settings (`--cuda-device` or `--hip-device`) are available.

## Usage

```bash
just build tnl-benchmark-scan-cuda

# all value types, scan types, and configurations
./build/bin/tnl-benchmark-scan-cuda --log-file scan-$(hostname).log

# only double, inclusive scan, and selected configurations
./build/bin/tnl-benchmark-scan-cuda --value-type double --scan-type inclusive \
    --block-size-factors 1 --values-per-thread 11 13 15
```

The benchmark writes the measurements in the JSONL format to the log file (`<program>.log` by
default), together with the metadata of the GPU in `<program>.metadata.json`, so the results
from different GPUs can be compared. Building the benchmark takes a while, because each
configuration instantiates the lookback kernel for all value types and scan types.
