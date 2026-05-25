# TNL Benchmarks Architecture

## File structure

Each benchmark subdirectory contains:

- **README.md** file providing documentation for the benchmark.
- **Entry point header** (`tnl-benchmark-<name>.h`) — defines `configSetup()`, `main()`, and
  optionally precision/device resolution functions. This file is included by wrappers (same
  basename) for all device targets: `.cpp` for host, `.cu` for CUDA compiler, and `.hip` for
  HIP compiler.
- **Benchmark logic headers** — define templated `run_benchmark*()` functions, benchmark
  helpers, etc. They are included by the entry point header.
- **CMakeLists.txt** — registers the executable targets. The host-only `tnl-benchmark-<name>`
  executable is built always, `tnl-benchmark-<name>-cuda` and `tnl-benchmark-<name>-hip`
  are built conditionally when the CUDA/HIP build is enabled.

## Entry point pattern

The entry point header must define exactly these two top-level symbols:

```cpp
void configSetup( TNL::Config::ConfigDescription& config );
int  main( int argc, char* argv[] );
```

Helper resolution functions (e.g. `resolveDevice`, `resolvePrecision`) are optional and
follow the "Resolving runtime parameters to template parameters" section when needed.

### Includes

Follow the *include what you use* convention.
For devices, include `TNL/Devices/GPU.h` rather than `Devices/Cuda.h` and `Devices/Hip.h`:

```cpp
#include <TNL/Devices/GPU.h>    // NOT <TNL/Devices/Cuda.h>
#include <TNL/Devices/Host.h>
```

### Namespaces

Each benchmark can use one of the following patterns:

1. Benchmark-specific symbols placed in `namespace TNL::Benchmarks::NameOfTheBenchmark { ... }`
2. No benchmark-specific namespace and `using namespace TNL;` etc. at the global scope
3. No benchmark-specific namespace and no `using namespace` directives

### `configSetup`

```cpp
void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );

   config.addDelimiter( "<Name> benchmark settings:" );
   // ... benchmark-specific entries (example) ...
   config.addEntry< std::string >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );

   config.addEntry< std::string >( "device", "Device to run benchmarks on.", "all" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );

   config.addDelimiter( "Device settings:" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::GPU::configSetup( config );
   TNL::MPI::configSetup( config );  // optional (only when the benchmark uses MPI)

   // ... other optional sections for classes with configSetup member functions ...
}
```

Rules:

- **Parameter name**: `"device"` (singular), never `"devices"`.
- **Default value**: `"all"` unless the benchmark is GPU-only (then `"cuda"`) or
  host-only (then `"host"`).
- **Enum ordering**: `host`, then `cuda`, then `hip`, then `all` always.
- **Device configSetup**: Always call both `Devices::Host::configSetup` and
  `Devices::GPU::configSetup` (not `Cuda::configSetup`). `GPU` internally configures
  either CUDA or HIP based on the build.
- **Delimiter**: Use `"Device settings:"` for the Host/GPU configSetup block.

### `main`

```cpp
int
main( int argc, char* argv[] )
{
   // optional (when the benchmark uses MPI):
   TNL::MPI::ScopedInitializer mpi( argc, argv );

   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   // optional (when the benchmark uses MPI):
   if( ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   // init benchmark
   TNL::Benchmarks::Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   resolvePrecision( benchmark, parameters );

   return EXIT_SUCCESS;
}
```

Rules:

- **Concerns**: Parse command line first, then device setup, then init benchmark,
  finally run a single entry-point function.
- **Device setup**: Always call both `Devices::Host::setup` and `Devices::GPU::setup`
  (not `Cuda::setup`). Both are unconditional — `GPU::setup` returns `true` even in
  non-GPU builds.
- **Call next**: Pass `benchmark` and `parameters` container down to the next function.
  Do not pass individual extracted parameters unnecessarily.

## Resolving runtime parameters to template parameters

The function called from `main` selects a suitable runtime parameter that needs to
be resolved to a template parameter and calls the next function. For example:

```cpp
void
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto& precision = parameters.getParameter< std::string >( "precision" );

   if( precision == "all" || precision == "float" )
      run_benchmark< float >( benchmark, parameters );
   if( precision == "all" || precision == "double" )
      run_benchmark< double >( benchmark, parameters );
}
```

Rules:

- **Dispatch pattern**: Use the `if (all || X)` pattern — never `if/else if/else`
  with an error branch. This allows `"all"` to run all enum values sequentially.

### Device dispatch pattern

The device parameter should be dispatched last in the dispatch sequence, because the
code for different devices is typically non-symmetric whereas other parameters should
be handled the same way for all devices.
The `run_benchmark` function template gets the device string from `parameters` and
dispatches conditionally:

```cpp
template< typename PrecisionType >
void
run_benchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto& device = parameters.getParameter< std::string >( "device" );

   if( device == "host" || device == "all" )
      run_benchmark_host< PrecisionType >( benchmark );

#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" )
      run_benchmark_gpu< PrecisionType >( benchmark );
#endif
}
```

Rules:

- **Host path**: Guarded only by `device == "host" || device == "all"`. No preprocessor
  guard needed.
- **GPU path**: Guarded by BOTH `#if defined( __CUDACC__ ) || defined( __HIP__ )` (compile-time)
  AND `device == "cuda" || device == "hip" || device == "all"` (runtime). The runtime
  check must include `"hip"` alongside `"cuda"` so that HIP-built executables match the
  `"hip"` CLI value.
- **GPU template parameter**: Use `Devices::GPU` (not `Devices::Cuda`) inside the
  `#if defined( __CUDACC__ ) || defined( __HIP__ )` block. Since `Cuda` and `Hip` are
  aliases to `GPU`, using `GPU` directly makes the code backend-agnostic.
- **Sequential device**: Some benchmarks support `"sequential"`. It should be listed as
  an enum and checked alongside `"host"` when present.

## Metadata columns

Use consistent naming for benchmark metadata:

| Key           | Values                                         | Notes                                        |
|---------------|------------------------------------------------|----------------------------------------------|
| `device`      | `"sequential"`, `"host"`, `"cuda"`, `"hip"`    | Auto-injected by `Benchmark::time<Device>()` |
| `precision`   | `"float"`, `"double"`                          | From `getType<PrecisionType>()`              |
| `operation`   | `"PI"`, `"QR"`, `"GEM"`, etc.                  | The algorithm being benchmarked              |
| `matrix type` | `"SM"`, `"DM_CMO"`, `"DM_RMO"`                 | Matrix types (benchmark-specific)            |

Keys use space-separated lowercase words, never CamelCase, lowercase camelCase, or snake_case.

### Performer convention

The **performer** field in benchmark results identifies the algorithm or implementation,
not the device it runs on. Device information is automatically injected as the `"device"`
metadata column by `Benchmark::time<Device>()` via `getDeviceName<Device>()`.

Rules:

- **TNL implementations**: Use `"TNL"` as the performer.
- **External libraries**: Use the library name: `"Boost"`, `"Gunrock"`, `"MAGMA"`,
  `"cuBLAS"`, `"hipBLAS"`, `"CuSolverWrapper"`, `"BLAS"`, etc.
- **Algorithm variants**: Use the algorithm name: `"legacy"`, `"std::partial_sum"`,
  `"std::inclusive_scan"`, `"thrust::inclusive_scan"`, etc.
- **Data transfers**: Use `"host-to-device"` / `"device-to-host"`.
- **Never use device names as performers**: Do not pass `"host"`, `"cuda"`, `"sequential"`,
  `"CPU"`, or `"GPU"` as the performer string — these belong in the `"device"` metadata
  column.

### Utility

`src/TNL/Benchmarks/Devices.h` provides:

- `getDeviceName<Device>()` — returns the lowercase device name string
  (`"sequential"`, `"host"`, `"cuda"`, `"hip"`)
- `checkDevice<Device>(parameters)` — returns `false` if the user's `--device`
  selection excludes this device, used to skip benchmark invocations

`getDeviceName` is used internally by `Benchmark::time<Device>()` and benchmark
code should not need to call it directly — only `checkDevice` for correct dispatch.

## Runner script

The `tnl-run-benchmarks.py` script in `src/Tools/` drives all benchmarks via YAML or
TOML config files. See `benchmarks.example.yaml` and `benchmarks.example.toml` for
the config schema. The script auto-discovers benchmark executables from `bin_dir`
by filename prefix and passes parameters via CLI.
