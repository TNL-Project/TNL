# Template Numerical Library

![TNL logo](tnl-logo.png)

TNL is a collection of building blocks that facilitate the development of
efficient numerical solvers. It is implemented in C++ using modern programming
paradigms in order to provide flexible and user friendly interface. TNL provides
native support for modern hardware architectures such as multicore CPUs, GPUs,
and distributed systems, which can be managed via a unified interface.
Visit the main [TNL web page](https://tnl-project.org/) for details.

## Installation   {#installation}

TNL is a header-only library, so it can be used directly after fetching the
source code (header files) without the usual build step. However, TNL has some
dependencies and provides several optional components that may be built and
installed on your system.

In the following subsections, we review the available installation methods.

### System-wide installation on Arch Linux

If you have an Arch Linux system, you can install the [tnl-git](
https://aur.archlinux.org/packages/tnl-git) package from the AUR. This will
do a complete build of TNL including all optional components. The advantage
of this approach is that all installed files and dependencies are tracked
properly by the package manager.

See the [Arch User Repository](
https://wiki.archlinux.org/title/Arch_User_Repository) wiki page for details
on using the AUR.

### Manual installation to the user home directory

You can clone the git repository via HTTPS:

    git clone https://gitlab.com/tnl-project/tnl.git

or via SSH:

    git clone git@gitlab.com:tnl-project/tnl.git

Then execute the `install` script to copy the header files to the final
location (`~/.local/include` by default):

    cd tnl
    ./install

However, we also recommend to install at least the `tools` [optional
component](#optional-components):

    ./install tools

Finally, see [Environment variables](#environment-variables).

### Manual installation with CMake

You can clone the git repository via HTTPS:

    git clone https://gitlab.com/tnl-project/tnl.git

or via SSH:

    git clone git@gitlab.com:tnl-project/tnl.git

Change to the cloned directory as all following commands are expected to be
run from there:

    cd tnl

The procedure consists of the three usual steps: configure, build, install.

1. The [configure step][cmake-configure]
   generates a build configuration for a particular build system.
   We recommend to use [Ninja](https://ninja-build.org/), in which case the
   configure command looks as follows:

       cmake -B build -S . -G Ninja

   Alternatively, you can use a [CMake preset][CMake preset] to generate a
   build configuration based on a named collection of options. Presets are
   defined in two files in the project's root directory: `CMakePresets.json`
   (a project-wide file tracked in the git repository) and
   `CMakeUserPresets.json` (user's own presets). For example, to use the
   preset named `default`:

       cmake --preset default

   All available configure presets can be listed by running the
   `cmake --list-presets` command.

   In both cases, you can add additional options to the `cmake` command. Note
   that options specified on the command line take precedence over the options
   set in the preset. The most common option for the configure step is
   [-D][cmake -D], which defines a variable in the CMake cache.

   TNL has the following CMake options that can be set with the `-D` option:

   - `TNL_USE_CUDA` – Build with CUDA support (ON by default)
   - `TNL_USE_HIP` – Build with HIP support (ON by default)
   - `TNL_USE_OPENMP` – Build with OpenMP support (ON by default)
   - `TNL_USE_MPI` – Build with MPI support (ON by default)
   - `TNL_USE_GMP` – Build with GMP support (OFF by default)
   - `TNL_USE_SYSTEM_GTEST` – Use GTest installed in the local system and do
     not download the latest version (OFF by default)
   - `TNL_USE_CI_FLAGS` – Add additional compiler flags like `-Werror` that
     are enforced in CI builds (OFF by default)
   - `TNL_USE_MARCH_NATIVE_FLAG` – Add `-march=native` and `-mtune=native` to
     the list of compiler flags for the Release configuration (OFF by default)
   - `TNL_BUILD_COVERAGE` – Enable code coverage reports from unit tests (OFF
     by default)
   - `TNL_OFFLINE_BUILD` – Offline build (i.e. without downloading libraries
     such as GTest) (OFF by default)

   The most common native CMake variables are:
   - [CMAKE_BUILD_TYPE][CMAKE_BUILD_TYPE] for setting the build type
     (e.g. `RelWithDebInfo` which is set in the default preset)
   - [CMAKE_CUDA_ARCHITECTURES][CMAKE_CUDA_ARCHITECTURES] for setting the
     list of [CUDA GPU architectures][CUDA_ARCHITECTURES] (also known as
     [CUDA compute capability][CUDA compute capability]), e.g. set to `80` for
     a GPU with a compute capability of 8.0.
   - [CMAKE_HIP_ARCHITECTURES][CMAKE_HIP_ARCHITECTURES] for setting the list of
     [HIP GPU architectures][HIP_ARCHITECTURES].

2. The [build step][cmake-build] invokes the build system and produces the
   specified targets. For example, to build *all* targets in the project's
   build tree:

       cmake --build build --target all

   You can replace `all` in the previous command with any of the following
   *utility targets*:

   - `benchmarks` – Build all targets in the `src/Benchmarks` directory
   - `documentation` – Build code snippets and generate the documentation
   - `examples` – Build all targets in the `src/Examples` directory
   - `tools` – Build all targets in the `src/Tools` directory
   - `tests` – Build all unit tests in the `src/UnitTests` directory
   - `matrix-tests` – Build only unit tests in the `src/UnitTests/Matrices`
     directory
   - `non-matrix-tests` – Build unit tests in the `src/UnitTests` directory,
     except `src/UnitTests/Matrices`.

   If you want to run the unit tests, use the following command with the
   special `test` target:

       cmake --build build test

3. The [install step][cmake-install] copies the already built targets and
   static files to a destination in the system specified by the `--prefix`
   option. For example, to install TNL to the user home directory:

       cmake --install build --prefix ~/.local

   Alternatively, you can install only a specific component (this is needed
   if you did not build the `all` target). The names of the available
   components are the same as the names of the *utility targets* in the build
   step, plus the `headers` component, which installs only the C++ header
   files:

       cmake --install build --component headers --prefix ~/.local

Finally, some notes and tips for TNL developers:

- The configure step is typically run only once. When you need to change some
  value in CMake cache, only the options with *modified* values need to be
  specified.
- The build step is used most frequently and the install step is not needed
  in the development workflow.
- The [CMake Tools][CMake Tools] extension for VSCode/VSCodium shows convenient
  buttons for common actions in the status bar. The CMake Tools extension
  supports CMake presets and allows to select the aforementioned *utility
  targets* for the build step.

[cmake-configure]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#generate-a-project-buildsystem
[cmake -D]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-D
[cmake-build]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project
[cmake-install]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#install-a-project
[CMake preset]: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
[CMake Tools]: https://open-vsx.org/extension/ms-vscode/cmake-tools
[CMAKE_BUILD_TYPE]: https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html
[CMAKE_CUDA_ARCHITECTURES]: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
[CUDA_ARCHITECTURES]: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
[CUDA compute capability]: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
[CMAKE_HIP_ARCHITECTURES]: https://cmake.org/cmake/help/latest/variable/CMAKE_HIP_ARCHITECTURES.html
[HIP_ARCHITECTURES]: https://cmake.org/cmake/help/latest/prop_tgt/HIP_ARCHITECTURES.html

## Dependencies   {#dependencies}

In order to use TNL, you need to install a compatible compiler, a parallel
computing platform, and (optionally) some libraries.

- __Supported operating systems:__
  TNL is frequently tested on Linux where all features are supported.
  Additionally, macOS and Windows are partially supported with some features
  missing (most notably, CUDA and MPI parallelization). Note that support for
  macOS and Windows is not tested frequently and there might be various
  compatibility issues. The incompatibilities and known issues related to the
  Windows operating system are tracked in [GitLab issue](
  https://gitlab.com/tnl-project/tnl/-/issues/115).

- __Supported compilers:__
  You need a compiler which supports the [C++17](
  https://en.wikipedia.org/wiki/C%2B%2B17) standard, for example [GCC](
  https://gcc.gnu.org/) 8.0 or later or [Clang](http://clang.llvm.org/) 7 or
  later.

  The [Microsoft Visual C++ (MSVC)](https://en.wikipedia.org/wiki/MSVC) compiler
  is currently [not supported](https://gitlab.com/tnl-project/tnl/-/issues/115).
  Instead, we recommend to use the [Windows Subsystem for Linux (WSL)](
  https://learn.microsoft.com/en-us/windows/wsl/) or the [MSYS2](
  https://www.msys2.org/) platform for developing code with TNL on Windows.

- __Parallel computing platforms:__
  TNL can be used with one or more of the following platforms:
    - [OpenMP](https://en.wikipedia.org/wiki/OpenMP) -- for computations on
      shared-memory multiprocessor platforms.
    - [CUDA](https://docs.nvidia.com/cuda/index.html) 11.0 or later -- for
      computations on Nvidia GPUs.
    - [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) -- TNL can
      a library implementing the MPI-3 standard for distributed computing (e.g.
      [OpenMPI](https://www.open-mpi.org/)). For distributed CUDA computations,
      the library must be [CUDA-aware](
      https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/).

- __Libraries:__
  Various libraries are needed to enable optional features or enhance the
  functionality of some TNL components. Make sure that all relevant packages are
  installed and use the appropriate flags when compiling your project.

  <table>
  <tr><th>Library</th>
      <th>Affected components</th>
      <th>Compiler flags</th>
      <th>Notes</th>
  </tr>
  <tr><td> [zlib](http://zlib.net/) </td>
      <td> \ref TNL::Meshes::Readers "XML-based mesh readers" and \ref TNL::Meshes::Writers "writers" </td>
      <td> `-DHAVE_ZLIB -lz` </td>
      <td> </td>
  </tr>
  <tr><td> [TinyXML2](https://github.com/leethomason/tinyxml2/) </td>
      <td> \ref TNL::Meshes::Readers "XML-based mesh readers" </td>
      <td> `-DHAVE_TINYXML2 -ltinyxml2` </td>
      <td> If TinyXML2 is not found as a system library, CMake
           will download, compile and install TinyXML2 along with TNL. </td>
  </tr>
  <tr><td> [CGAL](https://github.com/CGAL/cgal/) </td>
      <td> Additional mesh ordering algorithms for `tnl-reorder-mesh` and `tnl-plot-mesh-ordering` </td>
      <td> `-DHAVE_CGAL` </td>
      <td> Only used for the compilation of these tools. </td>
  </tr>
  <tr><td> [Metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) </td>
      <td> `tnl-decompose-mesh` </td>
      <td> </td>
      <td> Only used for the compilation of the `tnl-decompose-mesh` tool. </td>
  </tr>
  <tr><td> [Hypre](https://github.com/hypre-space/hypre) </td>
      <td> \ref Hypre "Wrappers for Hypre solvers" </td>
      <td> `-DHAVE_HYPRE -lHYPRE` </td>
      <td> Attention should be paid to Hypre build options, e.g. `--enable-bigint`. </td>
  </tr>
  </table>

- __Other language toolchains/interpreters:__
    - Python – install an interpreter for using the Python scripts included in
      TNL.

### Optional components   {#optional-components}

TNL provides several optional components such as pre-processing and
post-processing tools which can be compiled and installed by the `install`
script to the user home directory (`~/.local/` by default). The script can be
used as follows:

    ./install [options] [list of targets]

In the above, `[list of targets]` should be replaced with a space-separated list
of targets that can be selected from the following list:

- `all`: Special target which includes all other targets.
- `benchmarks`: Compile the `src/Benchmarks` directory.
- `documentation`: Compile code snippets and generate the documentation.
- `examples`: Compile the `src/Examples` directory.
- `tools`: Compile the `src/Tools` directory.
- `tests`: Compile unit tests in the `src/UnitTests` directory.
- `matrix-tests`: Compile unit tests in the `src/UnitTests/Matrices` directory.
- `non-matrix-tests`: Compile unit tests in the `src/UnitTests` directory,
  except `src/UnitTests/Matrices`.

Additionally, `[options]` can be replaced with a list of options with the `--`
prefix that can be viewed by running `./install --help`.

## Usage in other projects  {#usage}

To use TNL in another project, you need to make sure that TNL header files are
available and configure your build system accordingly. To obtain TNL, you can
either [install it](#installation) as described above, or add it as a git
submodule in your project as described in the next section. The last two
sections below provide examples for the configuration in CMake and Makefile
projects.

### Adding a git submodule to another project

To include TNL as a git submodule in another project, e.g. in the `libs/tnl`
location, execute the following command in the git repository:

    git submodule add https://gitlab.com/tnl-project/tnl.git libs/tnl

See the [git submodules tutorial](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
for details.

You will need to adjust the build system of your project to use TNL from the
submodule. The [Usage](#usage) section for some hints.

### CMake projects

There are two ways to incorporate TNL in a [CMake](https://cmake.org/)-based project:

1. Install TNL system-wide or in your user home directory where CMake can find
   it, and use `find_package(TNL)` in your project.
2. Add a git submodule for TNL to your project and include it with
   `add_subdirectory(libs/tnl)` in the `CMakeLists.txt` file.

See the [example projects](https://gitlab.com/tnl-project/example-projects) for details.

### Makefile projects

To incorporate TNL into an existing project using [GNU Make](https://www.gnu.org/software/make/)
as the build system, see the `Makefile` and `config.mk` files in the relevant
[example project](https://gitlab.com/tnl-project/example-projects/makefile).
The compiler flags used in the example project are explained in the
[Compiler flags](#compiler-flags) section.

## Tips and tricks

### Wrapper tnlcxx

`tnlcxx` is a wrapper which configures the build system (CMake) for simple situations where
the user needs to compile only one `.cpp` or `.cu` source file. The wrapper is available in a
separate [git repository](https://gitlab.com/tnl-project/tnlcxx).

### Compiler flags  {#compiler-flags}

Note that if you use TNL in a CMake project as suggested above, all necessary
flags are imported from the TNL project and you do not need to specify them
manually.

- Enable the C++17 standard: `-std=c++17`
- Configure the include path: `-I /path/to/include`
    - If you installed TNL with the install script, the include path is
      `<prefix>/include`, where `<prefix>` is the installation path (it is
      `~/.local` by default).
    - If you want to include from the git repository directly, you need to
      specify `<git_repo>/src` as an include paths, where `<git_repo>` is the
      path where you have cloned the TNL git repository. This may be a git
      submodule in your own project.
- Enable optimizations: `-O3 -DNDEBUG` (you can also add
  `-march=native -mtune=native` to enable CPU-specific optimizations).

Parallel computing platforms in TNL may be enabled automatically when using the
appropriate compiler, or additional compiler flags may be needed.

- CUDA support is automatically enabled when the `nvcc` or `clang++` compiler
  is used to compile a `.cu` file. This is detected by the `__CUDACC__`
  proprocessor macro.
    - For `nvcc`, the following flags are also required:
      `--expt-relaxed-constexpr --extended-lambda`
- OpenMP support must be enabled by defining the `HAVE_OPENMP` preprocessor
  macro (e.g. with `-D HAVE_OPENMP`). Also `-fopenmp` is usually needed to
  enable OpenMP support in the compiler.
- MPI support must be enabled by defining the `HAVE_MPI` preprocessor macro
  (e.g. with `-D HAVE_MPI`). Use a compiler wrapper such as `mpicxx` or link
  manually against the MPI libraries.

Of course, there are many other useful compiler flags. For example, the
flags that we use when developing TNL can be found in the
[cxx_flags.cmake][cxx_flags.cmake] and [cuda_flags.cmake][cuda_flags.cmake]
files in the Git repository.

[cxx_flags.cmake]: https://gitlab.com/tnl-project/tnl/-/blob/main/cmake/cxx_flags.cmake
[cuda_flags.cmake]: https://gitlab.com/tnl-project/tnl/-/blob/main/cmake/cuda_flags.cmake

### Environment variables   {#environment-variables}

If you installed some TNL tools or examples using the `install` script, we
recommend you to configure several environment variables for convenience. If you
used the default installation path `~/.local/`:

    export PATH="$PATH:$HOME/.local/bin"

These commands can be added to the initialization scripts of your favourite
shell, e.g. `.bash_profile`.
