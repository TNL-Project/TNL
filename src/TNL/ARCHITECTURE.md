# TNL library architecture

The following sections document design decisions of the TNL library.
It is relevant for TNL developers rather than users.

## CUDA-HIP portability

TNL targets both NVIDIA (CUDA) and AMD (HIP/ROCm) GPUs through shared GPU source files.
The following conventions ensure code compiles and runs correctly on both platforms.

### Device type dispatch

Prefer `Devices::GPU` over `Devices::Cuda || Devices::Hip` in `if constexpr` checks:

```cpp
// preferred
if constexpr( std::is_same_v< Device, Devices::GPU > ) { ... }

// avoid when both paths are identical
if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) { ... }
```

There should be no case where the code path is different for each backend.

### Warp size

NVIDIA GPUs always use warp size 32.
AMD GPUs have variable wavefront size: 64 on gfx8/gfx9 (GCN, CDNA), 32 on gfx10+ (RDNA).
Furthermore, AMD does not provide a way to get the wavefront size at compile-time.
The `__AMDGCN_WAVEFRONT_SIZE__` macro that previously provided a compile-time wavefront size was deprecated in ROCm 7.0 and removed in ROCm 7.2
(see [release notes](https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html#amdgpu-wavefront-size-macro-removal)).
This creates a host/device compilation split on HIP because architecture macros like `__GFX8__` and `__GFX9__` are defined **only during device compilation** — host code never sees them.

Use the warp-size API from `<TNL/Backend/LaunchHelpers.h>`:

| Function                | Scope         | Return value                                  |
|-------------------------|---------------|-----------------------------------------------|
| `getWarpSize()`         | Device only   | Actual warp size for the target architecture  |
| `getMaxWarpSize()`      | Host + device | Maximum across all architectures in the build |
| `getMinWarpSize()`      | Host + device | Always 32                                     |
| `getWarpSize(deviceId)` | Host only     | Runtime query via `hipDeviceGetAttribute`     |

Rules:

- **Device code**: use `getWarpSize()` or `getWarpFullMask()` freely.
- **Host-side compile-time guards**: use `getMaxWarpSize()`.
  It returns 64 on any HIP build (covering all AMD architectures in a fat binary) and 32 on CUDA.
- **Host-side runtime decisions**: use `getWarpSize(deviceId)` to get the actual device wavefront size at runtime.
- **Compile-time floors**: use `getMinWarpSize()` (always 32) for constraints that must allow configs valid on any architecture,
  e.g. column-major SlicedEllpack launch configs where `TPS * SliceSize >= getMinWarpSize()`.

The typical pattern for guard-plus-runtime-dispatch is:

```cpp
if constexpr( Backend::getMaxWarpSize() == 32 ) {
   // CUDA build: only TPS=32 path exists
   launchKernel<TPS=32>( ... );
}
else {
   // HIP build: both TPS=32 and TPS=64 kernels are instantiated
   if( Backend::getWarpSize( Backend::getDevice() ) == 32 )
      launchKernel<TPS=32>( ... );
   else
      launchKernel<TPS=64>( ... );
}
```

This ensures TPS=64 kernel instantiations exist in the HIP fat binary for wave64 devices, while CUDA builds pay zero binary bloat.

### Warp synchronization

Use cooperative groups instead of `__syncwarp()`, which is not available on HIP.
TNL provides a namespace alias in `<TNL/Backend/Functions.h>`:

```cpp
namespace TNL {
namespace cg = cooperative_groups;
}
```

Inside GPU kernels:

```cpp
auto warp = cg::tiled_partition< WarpSize >( cg::this_thread_block() );
warp.sync();

auto tile = cg::tiled_partition< GroupSize >( cg::this_thread_block() );
tile.sync();
```

### Shuffle instructions

HIP shuffle intrinsics (`__shfl_down_sync` etc.) differ from CUDA in two ways:

1. **All threads in the thread block must participate** in every shuffle call.
   On CUDA, only the threads selected by the mask argument need to execute the instruction — inactive lanes can take a divergent path.
   On HIP, divergence around a shuffle causes undefined behavior or deadlocks because AMD hardware executes warps in lockstep within a wavefront and the shuffle is a wavefront-wide operation.

2. **The mask argument is 64-bit** even on wave32 hardware (unused upper bits are zero).
   CUDA uses a 32-bit mask.
   Use `Backend::getWarpFullMask()` which returns the correct type and value for the target architecture.
   Do **not** call it from host code — it relies on `__GFX8__`/`__GFX9__` macros that are undefined on the host.

### Backend-safe API calls

Wrap all CUDA/HIP runtime API calls with `TNL_BACKEND_SAFE_CALL(call)`.
This maps to `cudaGetErrorString`/`hipGetErrorString` and throws `BackendRuntimeError` on failure.
