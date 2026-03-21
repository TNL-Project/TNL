# TNL Performance Rules

Practical rules to keep TNL code fast and portable across CPU, CUDA, and HIP backends.

## General Priorities
- Correctness and reproducibility first; optimize after behavior is validated.
- Prefer algorithmic improvements over micro-optimizations; benchmark before and after changes.

## Memory and Data Movement
- Minimize host↔device transfers; operate in-place on device-resident data when possible.
- Use views to avoid copying and to pass lightweight descriptors into kernels.
- Reuse scratch buffers instead of allocating per call; cache smart pointers/handles when reuse is expected.
- Keep headers lean to reduce compile-time and avoid accidental heavy copies.

## Kernel Design (CUDA/HIP)
- Keep kernel arguments small (views, sizes, simple functors); avoid capturing heavy objects or distributed containers.
- Avoid atomics unless necessary; prefer reduction/scan patterns that aggregate locally before global writes.
- Watch register pressure: simplify lambdas/functors and avoid unnecessary temporaries.
- Avoid divergent branches in hot loops; hoist invariant computations outside kernels when possible.

## Parallel Algorithm Choices
- Use provided primitives (`parallelFor`, `reduce`, `scan`, `sort`) instead of custom loops; they are tuned per backend.
- Choose stable vs. unstable sorting intentionally; stable sorts cost more.
- For reductions on floating point, be aware of order sensitivity; document nondeterminism and consider compensated summation if needed.

## CPU Performance
- Enable OpenMP when appropriate; keep work chunks reasonably sized to avoid overhead.
- Avoid excessive heap allocations inside tight loops; preallocate and reuse buffers.
- Keep cache locality in mind: contiguous access patterns for arrays/vectors and SoA-style layouts where applicable.

## MPI and Distribution (when used)
- Separate distributed concerns from device kernels; kernels should work on local partitions.
- Minimize collective frequency; batch communications and overlap with computation when possible.

## Build and Flags
- Respect CMake options: `TNL_USE_CI_FLAGS` for stricter warnings, `TNL_USE_MARCH_NATIVE_FLAG` for CPU tuning (optional), coverage flags guarded by `TNL_BUILD_COVERAGE`.
- Do not assume `-march=native`; keep code portable unless the option is enabled.

## Benchmarking and Validation
- Place performance probes in benchmarks under [src/Benchmarks](src/Benchmarks) or tooling under [src/Tools](src/Tools).
- Benchmark representative workloads on each targeted backend; record configs (device, flags, problem sizes).
- Use deterministic seeds and fixed inputs for reproducible measurements when feasible.

## Code Hygiene for Performance
- Preserve const-correctness and `noexcept` where trivial to help optimizers.
- Avoid unnecessary virtual dispatch in hot paths; favor templates and static polymorphism where appropriate.
- Reduce transitive includes and avoid ODR bloat by keeping implementations in TU-suitable locations when not header-only.
