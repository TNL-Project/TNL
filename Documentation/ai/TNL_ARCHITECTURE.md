# TNL Architecture Overview

This document sketches how the Template Numerical Library (TNL) is organized so AI-authored changes fit the library’s mental model.

## Layering
- **Core interfaces:** Header-only targets under [src/TNL](src/TNL) define allocators, devices, containers, algorithms, math utilities, and traits. Keep cross-cutting concepts here.
- **Execution backends:** `Devices` specialize execution for CPU, CUDA, and HIP; `Allocators` manage host/GPU/unified memory. Backend toggles are set via CMake options (`TNL_USE_CUDA`, `TNL_USE_HIP`, `TNL_USE_OPENMP`, `TNL_USE_MPI`).
- **Algorithms:** Parallel primitives (`parallelFor`, `reduce`, `scan`, `sort`) abstract execution; higher-level algorithms (solvers, grid/mesh ops) build on them.
- **Containers & Views:** `Array`, `Vector`, `NDArray`, matrix/segment/mesh structures. Views provide shallow, fixed-size access for kernels and lambdas.
- **Numerical stacks:** Linear/ODE solvers, grids/meshes, sparse formats, and supporting utilities compose the primitives and containers.
- **Ecosystem:** Examples, benchmarks, tools, and Python utilities demonstrate usage and provide profiling hooks; documentation (Users’ Guide, Doxygen) explains public contracts.

## Design Tenets
- **Portability first:** All public APIs should be usable across enabled backends; avoid host-only assumptions in shared code paths.
- **Value semantics & zero-cost abstractions:** Minimize hidden allocations; prefer views for passing data to kernels.
- **Explicit ownership:** Use RAII and clear allocator/device propagation; avoid implicit transfers.
- **Determinism:** Favor reproducible algorithms; document non-deterministic behavior when unavoidable.

## CMake/Build Integration
- Core target `TNL` is an interface library supplying headers, compile features, and backend flags.
- Optional dependencies are fetched via CMake modules (TinyXML2, GTest, etc.) with system-first preference.
- Testing is wired through `cmake/testing.cmake`; documentation via Doxygen configs in [Documentation](Documentation).

## Extension Points
- **New containers/algorithms:** Add under [src/TNL](src/TNL) with device-parameterized interfaces and view support.
- **Backend-specific kernels:** Isolate device code; guard with macros and keep host fallbacks aligned.
- **Examples/tests:** Mirror production patterns in [src/Examples](src/Examples) and [src/UnitTests](src/UnitTests) to illustrate and validate new APIs.
- **Docs:** Extend Users’ Guide sections and Doxygen pages for new public concepts.

## Dependency Discipline
- Prefer standard library and existing TNL utilities; introduce third-party code only through CMake modules with opt-in switches.
- Keep headers light: avoid heavy transitive includes; use forward declarations and traits to reduce compile times.

## Error Handling and Contracts
- Use assertions for internal invariants (`Assert`); validate user inputs where cheap.
- Maintain const-correctness and noexcept where obvious; document preconditions and side effects in Doxygen.
