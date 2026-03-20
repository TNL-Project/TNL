# Template Numerical Library (TNL)

## Project overview

TNL (Template Numerical Library) is a C++17 library for developing efficient numerical solvers with native support for modern parallel architectures. It provides a unified interface for multi-core CPUs (OpenMP), NVIDIA GPUs (CUDA), and AMD GPUs (HIP), along with MPI support for distributed systems.

**Key characteristics:**

- **Language**: C++17 with CUDA/HIP extensions
- **Build system**: CMake 3.30+ with Ninja generator
- **Target platforms**:
  - Linux (Arch Linux in CI), with support for GCC, Clang, NVHPC, and CUDA/HIP toolchains
  - macOS (not tested in CI)
- **License**: MIT

**Project structure:**

- `src/TNL/` - Core library headers (template-based, header-only)
- `src/Examples/` - Stand-alone examples
- `src/Benchmarks/` - Performance benchmarks
- `src/Tools/` - Pre/post-processing utilities
- `src/Python/` - Python utilities and wrappers
- `Documentation/` - Documentation pages (Doxygen), examples, users' guide
- `cmake/` - CMake modules and configuration
- `scripts/` - Utility scripts

## Building and running

### Prerequisites

- CMake 3.30+
- C++17 compiler (GCC, Clang, or NVHPC)
- CUDA toolkit (optional, for GPU support)
- ROCm/HIP (optional, for AMD GPU support)
- OpenMP (optional)
- MPI (optional)

### Quick start with `just`

The project uses [just](https://github.com/casey/just) as a command runner.
Key recipes:

```bash
# Configure with default options (defined in .env)
just configure

# Build th project (all targets by default)
just build

# Build specific targets
just build tests benchmarks

# Run all tests (also ensures that the built binaries are up to date)
just test

# Run checks (typos, formatting, linting, etc.)
just check

# Reformat all files
just format

# Clean the build directory
just clean
```

### CMake options

The CMake build system can be configured manually with custom options:

```bash
# Configure
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

Common build options:

| Option | Default | Description |
|--------|---------|-------------|
| `TNL_USE_CUDA` | ON | Enable CUDA support |
| `TNL_USE_HIP` | ON | Enable HIP support |
| `TNL_USE_OPENMP` | ON | Enable OpenMP support |
| `TNL_USE_MPI` | ON | Enable MPI support |
| `TNL_USE_CI_FLAGS` | OFF | Enable strict CI flags (`-Werror`) |

### CMake presets

The `CMakePresets.json` file also specifies several presets for different compilers (they all use `CMAKE_BUILD_TYPE=RelWithDebInfo`):

- `toolchain-gcc` - GCC host-only
- `toolchain-clang` - Clang host-only (with libc++)
- `toolchain-nvhpc-host` - NVHPC (nvc++) host-only
- `toolchain-gcc-nvcc` - GCC + CUDA (nvcc)
- `toolchain-clang-cuda` - Clang + CUDA
- `toolchain-hip` - ROCm/HIP

To use a preset, pass its name to `cmake --preset`:

```bash
cmake --preset toolchain-gcc`
```

## Development conventions

### Code formatting

- **Indentation**: 3 spaces for C++/CUDA/HIP, 4 spaces for Python/CMake, 2 spaces for YAML/.clangd
- **Line length**: 128 characters
- **Line endings**: LF (Unix)
- **Final newline**: Required

The project uses tools such as clang-format, gersemi, or ruff for automatic formatting.
They are automated with `just`, run `just format` to reformat all supported files
or `just check-format` (CI compatible) to check formatting compliance.

### Git best practices

**Branches:**

- The `main` branch is protected and cannot be pushed to directly
- Use feature branches named like `AB/some-feature`, where AB are the initials of the author
- Run `git config user.name` to get the author's full name

**Commits:**

- **Never run `git add -A` or `git add .`** - always stage files explicitly with
  `git add <filepath>` to avoid committing unintended changes (e.g., benchmark
  logs or script outputs)

**Commit message style:**

- **Subject line**: ≤72 characters, no period at end, imperative mood
- **Body**: ≤72 characters per line, blank line after subject
- **Content**: Describe what changed and why, not how
- **Attribution**: All AI-generated commits should have a `Co-authored-by:` attribution
  at the end, e.g. `Co-authored-by: qwen3.5 <noreply@none.ai>`
- **Workflow**: Human developers use rebase-based workflow and squash small commits.
  Hence, small commits that fix issues in previous commits should be marked accordingly.
  Create such commits with `git commit --squash=<commit-hash-to-squash-into>`.
  Such commits will be processed later by `git rebase --interactive --autosquash`.

### Coding style

- Follow existing code formatting (the "mirroring" principle)
- Run `clang-format`, `ruff format` or `gersemi` to ensure correct formatting of
  C++/CUDA/HIP files, Python files, or CMake files, respectively.
- Namespace indentation: None (flatten nested namespaces)
- Prefer range-based for loops, modern C++ features
- For CUDA and HIP, prefer using the facilities in `src/TNL/Backend/` over
  native C-like APIs, namely:
  - Use `__cuda_callable__` instead of separate `__host__`/`__device__`
  - Use the `TNL_BACKEND_SAFE_CALL` macro for checking low-level CUDA API calls

### Markdown style

- Use title-case in first-level headings only: `# Page Title`
- Use sentence-case in all other section levels: `## First section`, `### Some subsection`

### Testing

- **Framework**: Google Test (GTest) via CTest
- **Running**: `just test` or `ctest --preset all-tests`
- **Preset filters**: `matrix-tests`, `non-matrix-tests`, `mpi-tests`, `non-mpi-tests`

### Debugging

- **Important:** When nvcc is used for compiling CUDA code, the compiler errors
  must be resolved in a strictly sequential order from top to bottom. This is
  because *nvcc is infamous for producing nonsensical errors* that are in fact
  consequence of a previous error and it does not make sense to waste time on
  catching red herrings.
- When a compiled binary fails due to a CUDA error (backend runtime exception)
  or a TNL assertion error, run the binary through `compute-sanitizer` to get
  more insights.
- When a compiled binary fails due to an unspecified error (*segfault*), ensure
  that it is built with debugging info (`RelWithDebInfo` or `Debug` build type)
  and run the binary through `gdb` to get a backtrace.

### Linting and quality checks

```bash
# Full check suite
just check

# Individual checks:
just check-typos      # Spell checking (typos)
just check-format     # Code formatting (clang-format, ruff, gersemi)
just check-recipes    # justfile shell scripts (shellcheck)
just check-shellcheck # Shell scripts (shellcheck)
just check-python     # Python (ruff)
just check-clang-tidy # C++ (clang-tidy)
```

**Note**: clang-tidy requires a non-CUDA build (C++ only) in Debug mode.

## Directory structure details

### `src/TNL/` - Core library

Header-only templates organized by domain:

- `Containers/` - Array, Vector, Matrix views
- `Matrices/` - Sparse/dense matrix formats
- `Meshes/` - Unstructured mesh data structures
- `Solvers/` - Linear/nonlinear solvers, ODE/PDE solvers
- `Algorithms/` - Sorting, reduction, traversal algorithms
- `Backend/` - Abstraction layer for CPU/GPU backends
- `Devices/` - Device-specific implementations
- `MPI/` - MPI utilities and parallel algorithms
- `Config/` - Configuration macros and type traits

### `src/Examples/` - Examples

Demonstrative solvers showing library usage:

- `tnl-optimize-ranks` - Rank optimization example
- `tnl-turbulence-generator` - Turbulence simulation
- Both CPU (`.cpp`) and GPU (`.cu`) versions available

### `src/Benchmarks/` - Performance benchmarks

Benchmarks for measuring performance of core components.

- Each directory should contain a `README.md` file describing the benchmark
- Some benchmarks may be implemented for multiple devices, in which case the
  code must be generic and placed in a header file (`tnl-benchmark-<component>.h`)
  which is included from the corresponding `.cpp` and `.cu` files.
- Each benchmark must use the `TNL::Benchmarks::Benchmark` class for measuring
  and logging.

### `src/Tools/` - Utility tools

Command-line tools for data preprocessing and postprocessing.

## Important notes for AI assistants

1. **CUDA/HIP builds**: When working with GPU code, be aware that:
   - CUDA and HIP are mutually exclusive (can't build both simultaneously)
   - Some tools (clang-tidy) only work with CPU builds
   - Test filters exist to run only matrix/non-matrix tests

2. **Build artifacts**: Located in `build/` directory by default
   - `compile_commands.json` is generated for IDE/clang tool support
   - Executables go to `build/bin/`, libraries to `build/lib/`

3. **Installation**: TNL is a header-only library with some optional compiled components (tools, examples, benchmarks)

4. **CI/CD**: The project uses GitLab CI with Docker images for various toolchains (Arch Linux base)

5. **CMake best practices**: The project uses modern CMake with:
   - Interface targets (`TNL::TNL`)
   - Export sets for downstream projects
   - `FetchContent` for dependencies (GTest, TinyXML2)

6. **Python integration**: The `src/Python/` directory contains utilities; full Python bindings are in a separate `PyTNL` repository

## References

- [Main website](https://tnl-project.org/)
- [Documentation](https://tnl-project.gitlab.io/tnl/)
- [GitLab repository](https://gitlab.com/tnl-project/tnl)
- [Contributing guidelines](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
