# TNL Parallel Patterns

Reusable patterns for writing parallel code that stays portable across CPU, CUDA, and HIP backends.

## Core Primitives
- **parallelFor:** Use for elementwise and structured loops; template on `Device` and iterate with index lambdas. Prefer views in captures.
- **reduce:** For associative reductions; provide identity and functor carefully to ensure determinism where possible.
- **scan:** Inclusive/exclusive prefix operations; use for segmented tasks and index construction.
- **sort:** Device-aware sorting; prefer existing sorting utilities before rolling bespoke kernels.

## Container & View Usage
- Operate on `Array`/`Vector`/`NDArray` views inside kernels to avoid allocator overhead and to pass shallow copies to device lambdas.
- Keep views fixed-size; resize the owning container on host before launching kernels.

## Device Specialization
- Template algorithms on `Device` (CPU, CUDA, HIP) and `Value` types; avoid duplicated host/GPU code branches when templates suffice.
- Isolate backend-only pieces behind `if constexpr` on device traits or preprocessor guards driven by CMake options.
- Avoid host-only constructs in device lambdas (RTTI, iostream, mutexes, locale-heavy ops).

## Work Partitioning Guidelines
- Prefer flat parallelism via `parallelFor` unless data locality requires tiling; keep kernel bodies small to minimize register pressure.
- For reductions, use tree-based or block-level patterns provided by existing primitives; avoid ad-hoc atomics unless measured and justified.
- When building segmented operations, leverage scans to compute offsets and then scatter/gather through views.

## Synchronization & Ownership
- Avoid implicit synchronizations; structure code so host-device transfers are explicit and minimal.
- Do not capture heavy objects or distributed containers in kernels; pass local slices/views.
- Cache temporary buffers (e.g., scratch arrays) when invoking patterns repeatedly to reduce allocations.

## Determinism & Reproducibility
- Reductions on floating point can be order-dependent; document when results may differ across backends or runs.
- Use stable sort variants when order matters; otherwise prefer faster unstable variants.

## Testing Parallel Code
- Add unit tests per backend as available; guard tests with build options to avoid running unsupported backends.
- Cover edge cases: empty ranges, small sizes, non-power-of-two counts, and misaligned segment boundaries.

## When to Write a Custom Kernel
- Only if primitives cannot express the needed pattern efficiently.
- Keep kernel interfaces minimal: views, sizes, and lightweight functors.
- Measure against existing primitives; prefer improving primitives if gaps are general.
