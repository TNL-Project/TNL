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

- `src/TNL/` - Core library headers (template-based, header-only, `TNL::` namespace)
- `src/Examples/` - Stand-alone examples
- `src/Benchmarks/` - Performance benchmarks
- `src/Tools/` - Pre/post-processing utilities
- `src/Python/` - Python utilities and wrappers
- `src/UnitTests/` - Google Test-based test suite
- `Documentation/` - Documentation pages (Doxygen), examples included in the documentation, users' guide
- `cmake/` - CMake modules and configuration
- `scripts/` - Utility scripts

## Building and running

### Prerequisites

- CMake 3.30+
- C++17 compiler (GCC, Clang, or NVHPC)
- CUDA toolkit (optional)
- ROCm/HIP (optional)
- OpenMP (optional)
- MPI (optional)

### Quick start with `just`

The project uses [just](https://github.com/casey/just) as a command runner.
Key recipes:

```bash
just configure    # Configure with default options
just build        # Build the project (all targets by default)
just build tests benchmarks  # Build specific targets
just test         # Run all tests
just check        # Run checks (typos, formatting, linting, etc.)
just format       # Reformat all files
just clean        # Clean the build directory
```

### Single test execution

```bash
ctest -R DenseMatrixTest         # Run a single test by name
ctest -L Matrices                # Run tests by label
ctest --preset matrix-tests      # Using test presets
```

### CMake options

The CMake build system can be configured manually with custom options:

```bash
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

Available presets (all use `CMAKE_BUILD_TYPE=RelWithDebInfo`):

- `toolchain-gcc` - GCC host-only
- `toolchain-clang` - Clang host-only (with libc++)
- `toolchain-nvhpc-host` - NVHPC (nvc++) host-only
- `toolchain-gcc-nvcc` - GCC + CUDA (nvcc)
- `toolchain-clang-cuda` - Clang + CUDA
- `toolchain-hip` - ROCm/HIP

Usage: `cmake --preset toolchain-gcc` or `cmake --preset toolchain-gcc --fresh`

The `--fresh` option is preferred over removing the build directory manually.

## Development conventions

### Code formatting

- **Indentation**: 3 spaces for C++/CUDA/HIP, 4 spaces for Python/CMake, 2 spaces for YAML/.clangd
- **Line length**: 128 characters for C++, 88 for Python
- **Line endings**: LF (Unix)
- **Final newline**: Required

Use `just format` or `just check-format` for automatic formatting.

### Markdown style

- Use title-case in first-level headings only: `# Page Title`
- Use sentence-case in all other section levels: `## First section`, `### Some subsection`

### Git best practices

Use `git` for managing the local repository and `glab` for the remote repository on GitLab (issues, merge requests, CI, etc).

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
- **Attribution**: All AI-generated commits should have a `Assisted-by:` attribution at the end, e.g. `Assisted-by: Qwen 3.5 via Opencode`
- **Workflow**: Human developers use rebase-based workflow and squash small commits.
  Hence, small commits that fix issues in previous commits should be marked accordingly.
  Create such commits with `git commit --squash=<commit-hash-to-squash-into>`.
  Such commits will be processed later by `git rebase --interactive --autosquash`.

### Coding style

- Follow existing code formatting ("mirroring" principle)
- Namespace indentation: None (flatten nested namespaces)
- Prefer range-based for loops, modern C++ features
- Use portable facilities in `src/TNL/Backend/` for CUDA and HIP
- Use `__cuda_callable__` instead of separate `__host__`/`__device__`
- Use wrappers in `src/MPI/` for MPI operations
- Benchmarks must use the `TNL::Benchmarks::Benchmark` class for measuring and logging.

### Imports and headers

- Header guards: `#pragma once`
- Include order: standard library first, then third-party libraries, then project headers
- Prefer angle brackets for TNL headers
- Group and sort includes logically
- Avoid cyclic dependencies; use forward declarations when appropriate

### Error handling

- **TNL_ASSERT_* macros** for debugging (disabled in optimized builds with `NDEBUG`)
  - `TNL_ASSERT_TRUE`, `TNL_ASSERT_FALSE`, `TNL_ASSERT_EQ`, `TNL_ASSERT_NE`
  - `TNL_ASSERT_LT`, `TNL_ASSERT_LE`, `TNL_ASSERT_GT`, `TNL_ASSERT_GE`
- **TNL_BACKEND_SAFE_CALL(call)** for low-level CUDA/HIP API calls
- **TNL::Exceptions::BackendRuntimeError** for runtime errors

### Testing

- **Framework**: Google Test (GTest) via CTest
- **Test structure**: Tests in `src/UnitTests/` subdirectories, each with its own `CMakeLists.txt`
- **Test labels**: `Matrices`, `MPI`, etc.
- **File extensions**: `.cpp`, `.cu` (CUDA), `.hip` (HIP)

### Debugging

- **CUDA compiler errors**: Resolve `nvcc` errors sequentially (top to bottom) - `nvcc` produces cascading errors
- **Runtime errors**: Use `compute-sanitizer` for CUDA/HIP errors or assertion failures
- **Segfaults**: Build with `RelWithDebInfo` or `Debug`, then run through `gdb`

### Linting and quality checks

```bash
just check              # Full check suite
just check-typos        # Spell checking
just check-format       # Code formatting (clang-format, ruff, gersemi)
just check-recipes      # justfile shell scripts
just check-shellcheck   # Shell scripts
just check-python       # Python (ruff)
just check-clang-tidy   # C++ (clang-tidy)
```

**Note**: clang-tidy requires a non-CUDA build in Debug mode.

## Running CI jobs locally

[gitlab-ci-local](https://github.com/firecow/gitlab-ci-local) can be used to run GitLab CI pipelines locally using podman or docker.

### Configuration

The project uses `.gitlab-ci-local-env` for default options:

```env
EVALUATE_RULE_CHANGES=false
```

This ignores `rules:changes` so all jobs are available regardless of file changes.

### Usage examples

```bash
# List available jobs
gitlab-ci-local --list
gitlab-ci-local --list-all         # include when:never jobs

# Run all jobs in a stage
gitlab-ci-local --stage lint --concurrency 4
gitlab-ci-local --stage build:host --concurrency 2
gitlab-ci-local --stage build:cuda --concurrency 1
gitlab-ci-local --stage build:hip_rocm --concurrency 1

# Run a specific job
gitlab-ci-local "ruff format"
gitlab-ci-local "gcc: [tests,Debug]"

# Run CUDA jobs with NVIDIA GPU passthrough (requires NVIDIA CDI)
gitlab-ci-local --device nvidia.com/gpu=all "nvcc: [tests,Debug]"

# Run HIP/ROCm jobs with AMD GPU passthrough
gitlab-ci-local --device /dev/kfd --device /dev/dri "hip_rocm: [tests,Debug]"

# Verify GPU passthrough with dummy check jobs
gitlab-ci-local --device nvidia.com/gpu=all "cuda-gpu-check"
gitlab-ci-local --device /dev/kfd --device /dev/dri "rocm-gpu-check"

# Override matrix variables
gitlab-ci-local --variable MATRIX_TARGET=tests --variable BUILD_TYPE=Release gcc

# Force shell executor instead of containers
gitlab-ci-local --force-shell-executor "typos"
```

### GPU passthrough

- **NVIDIA**: `--device nvidia.com/gpu=all` (via CDI, no `--privileged` needed)
- **AMD**: `--device /dev/kfd --device /dev/dri` (CDI does not cover ROCm)

### Key options

| Option | Description |
|--------|-------------|
| `--container-executable podman` | Use podman instead of docker |
| `--evaluate-rule-changes false` | Ignore `rules:changes` (run all jobs) |
| `--concurrency N` | Limit parallel jobs |
| `--device nvidia.com/gpu=all` | NVIDIA GPU passthrough via CDI |
| `--device /dev/kfd` | AMD KFD device for ROCm |
| `--device /dev/dri` | AMD DRI render nodes for ROCm |
| `--stage <name>` | Run all jobs in a stage |
| `--force-shell-executor` | Run on host without containers |
| `--shell-isolation` | Enable artifact isolation for shell jobs |
| `--variable KEY=VAL` | Set CI variables |
| `--pull-policy if-not-present` | Avoid re-pulling images |

### Caveats

- **Untracked files are not synced** into job containers — only git-tracked files. Run `git add` on new files before executing.
- **Per-stage concurrency is not supported** — run each stage separately with different `--concurrency` values.
- **NVIDIA GPU passthrough** requires `nvidia-ctk` CDI setup; use `--device nvidia.com/gpu=all`.
- **AMD GPU passthrough** requires `--device /dev/kfd --device /dev/dri` (CDI does not cover ROCm).
- **`rules:changes` may exclude jobs** — use `--evaluate-rule-changes false` to run all jobs regardless.
- **`artifacts`, `needs`, and `pages`** features are not fully supported by local execution.

## Important notes

- **CUDA/HIP builds**: CUDA and HIP are mutually exclusive; some tools only work with CPU builds
- **Build artifacts**: Located in `build/` directory; executables are in `build/bin/` (flat layout); `compile_commands.json` for LSP support
- **Installation**: Header-only library with optional compiled components
- **CMake best practices**: Use modern CMake with interface targets (`TNL::TNL`), export sets, and `FetchContent` for dependencies

## References

- [Main website](https://tnl-project.org/)
- [Documentation](https://tnl-project.gitlab.io/tnl/)
- [GitLab repository](https://gitlab.com/tnl-project/tnl)
- [Contributing guidelines](CONTRIBUTING.md)
- [Unit tests architecture](src/UnitTests/ARCHITECTURE.md)
