# TNL AI Developer Guide

This guide distills project rules and patterns to help AI assistants produce code and reviews that fit the Template Numerical Library (TNL).

## Core Principles
- Favor correctness, determinism, and reproducibility before micro-optimizations.
- Keep portability to CPUs, CUDA, HIP, MPI, and OpenMP; avoid assumptions about a single backend.
- Maintain readability: mirror existing patterns when style is unclear.
- Prefer zero-cost abstractions and value semantics; avoid hidden allocations or ownership surprises.
- Prefer `TNL::StaticArray` over C-style fixed-size arrays to align with project abstractions and device portability.

## Language, Build, and Options
- TNL builds as a C++17 header-only interface target defined in [CMakeLists.txt](CMakeLists.txt). Keep new code compatible with C++17.
- Respect optional backends toggled via CMake options: `TNL_USE_CUDA`, `TNL_USE_HIP`, `TNL_USE_OPENMP`, `TNL_USE_MPI`, `TNL_USE_GMP`, `TNL_USE_CI_FLAGS`, `TNL_USE_MARCH_NATIVE_FLAG`, and `TNL_BUILD_COVERAGE`. Guard backend-specific code with the corresponding macros and CMake conditionals.
- CUDA/HIP kernels must compile with separable compilation; keep device code free of host-only constructs.
- Prefer adding features through the existing CMake options and interface targets rather than ad-hoc flags.

## Formatting and Linting
- Use project formatting: `.editorconfig` for basics, `.clang-format` via `scripts/run-clang-format.py` for C++ (handles CUDA launches and OpenMP pragmas).
- Keep clang-tidy clean; config lives in `.clang-tidy` and runs in CI.
- Apply formatting before committing; avoid style-only follow-up commits.

## Naming Conventions
- Namespaces and directories: keep lowercase, mirroring existing structure (e.g., `TNL::Containers`, files under `src/TNL/Containers`).
- Types, classes, structs, enums: PascalCase (e.g., `Array`, `DistributedVector`).
- Functions and methods: camelCase verbs (e.g., `setSize`, `getView`, `parallelFor`). Prefer descriptive names over abbreviations.
- Template parameters: clear, capitalized nouns (`Device`, `Index`, `Real`, `Allocator`).
- Constants and enum values: PascalCase unless aligning with existing uppercase macros; avoid ALL_CAPS except for macros and compile-time flags.
- Macros: ALL_CAPS with underscores, prefixed when practical to avoid collisions.
- Member variables: mirror prevailing style in the touched file; prefer self-explanatory names and avoid Hungarian notation.
- File names: match primary type or module, using CamelCase for public headers under `src/TNL` (e.g., `Vector.h`, `ArrayView.h`).

## Repository Layout (high level)
- Core headers: [src/TNL](src/TNL) — algorithms, containers, devices, allocators, math utilities.
- Examples and benchmarks: [src/Examples](src/Examples) and [src/Benchmarks](src/Benchmarks) — mirror production usage; follow these patterns when adding samples.
- Tools: [src/Tools](src/Tools) — preprocessing/postprocessing utilities.
- Python utilities: [src/Python](src/Python); bindings live in external PyTNL repo.
- Documentation: [Documentation](Documentation) — Doxygen config, Users' Guide, and pages; update relevant pages when adding user-facing features.

## Core Architectural Concepts (from documentation)
- Memory space vs. execution model: allocation handled by allocators; execution specialized by devices/executors.
- Algorithms are templated on `Device` (e.g., `parallelFor`, `reduce`, `scan`, `sort`) to run on host or accelerator.
- Containers (Array, Vector, NDArray, etc.) abstract data management across devices.
- Views wrap raw data plus metadata; they are shallow-copyable and fixed-size. Use views in kernels and device lambdas to avoid costly copies.

## Design Guidelines for New Code
- Make APIs device-agnostic: template on `Device` and value types instead of duplicating host/GPU code.
- Keep CUDA/HIP kernels lightweight: avoid passing heavy objects (smart pointers, distributed containers); operate on local parts and views.
- Avoid distributed-object members inside kernels; MPI concerns belong above device-level kernels.
- Cache smart pointers or expensive handles when reuse is expected to reduce allocations and copies.
- Use existing parallel primitives (`parallelFor`, `reduce`, `scan`, `sort`) instead of ad-hoc loops.
- Preserve const-correctness, noexcept where trivial, and prefer RAII over manual resource management.
- Prefer standard library facilities first; add third-party deps only through CMake modules.

## Testing Expectations
- Prefer GoogleTest-based unit tests under [src/UnitTests](src/UnitTests) (fetched via CMake). Cover new public APIs and edge cases for each backend you support.
- Keep tests deterministic and backend-guarded (enable/disable with build options).
- Use CTest integration provided in [cmake/testing.cmake](cmake/testing.cmake) to register tests.

## Documentation Expectations
- Update or add Doxygen comments for new public symbols; keep terminology consistent with the Users' Guide and Pages (e.g., allocators, devices, views).
- Extend the Users' Guide or relevant page when introducing new concepts or behaviors.
- Provide small runnable examples under [src/Examples](src/Examples) when adding user-facing algorithms or containers.

## Git and Workflow (summarized from contributing guide)
- Configure git user name/email to match your GitLab profile.
- Write commits with a short (≤50 char) subject without trailing period; body wrapped at ~72 chars with rationale and context.
- Split changes into logical commits; use `git add -p` to stage granularly.
- Follow rebase-based workflow on feature branches; rebase frequently on `develop`, squash fixup commits before review, and open merge requests with WIP tagging when appropriate.

## How to Prompt the AI for Code Changes
- Specify target files, devices/backends affected, and expected behavior across CPU/CUDA/HIP/MPI.
- Ask for changes in terms of existing abstractions (allocators, devices, views, parallel primitives) and prefer extending existing modules in [src/TNL](src/TNL) rather than introducing new patterns.
- Request formatting via the project scripts and lint cleanliness when applicable.
- Include testing expectations: which unit tests to add or extend, and which backends must be covered.
- Mention documentation updates needed (Doxygen, Users' Guide section, example programs).
- Provide build configuration (CMake options toggled) and performance/portability constraints.

## Quick Checklist for AI-Authored Patches
- Matches project formatting and passes clang-tidy.
- Uses views and device-templated algorithms; no heavy objects inside kernels.
- Guards backend-specific code with existing CMake options/macros.
- Adds or updates tests and docs alongside code changes.
- Keeps commits small, well-titled, and rebase-ready.
