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

In the following, we review the available installation methods:

1. __System-wide installation on Arch Linux__

   If you have an Arch Linux system, you can install the [tnl-git](
   https://aur.archlinux.org/packages/tnl-git) package from the AUR. This will
   do a complete build of TNL including all optional components. The advantage
   of this approach is that all installed files and dependencies are tracked
   properly by the package manager.

   See the [Arch User Repository](
   https://wiki.archlinux.org/title/Arch_User_Repository) wiki page for details
   on using the AUR.

2. __Manual installation to the user home directory__

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

   Finally, see [Environment variables](#environment-variables)

3. __Adding a git submodule to another project__

   To include TNL as a git submodule in another project, e.g. in the `libs/tnl`
   location, execute the following command in the git repository:

       git submodule add https://gitlab.com/tnl-project/tnl.git libs/tnl

   See the [git submodules tutorial](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
   for details.

   You will need to adjust the build system of your project to use TNL from the
   submodule. The [Usage](#usage) section for some hints.

### Dependencies   {#dependencies}

In order to use TNL, you need to install a compatible compiler, a parallel
computing platform, and (optionally) some libraries.

- __Supported compilers:__
  You need a compiler which supports the [C++17](
  https://en.wikipedia.org/wiki/C%2B%2B17) standard, for example [GCC](
  https://gcc.gnu.org/) 8.0 or later or [Clang](http://clang.llvm.org/) 7 or
  later.

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
  <tr><td> [libpng](http://www.libpng.org/pub/png/libpng.html) </td>
      <td> \ref TNL::Images "Image processing" classes </td>
      <td> `-DHAVE_PNG_H -lpng` </td>
      <td> </td>
  </tr>
  <tr><td> [libjpeg](http://libjpeg.sourceforge.net/) </td>
      <td> \ref TNL::Images "Image processing" classes </td>
      <td> `-DHAVE_JPEG_H -ljpeg` </td>
      <td> </td>
  </tr>
  <tr><td> [DCMTK](http://dicom.offis.de/dcmtk.php.en) </td>
      <td> \ref TNL::Images "Image processing" classes </td>
      <td> `-DHAVE_DCMTK_H -ldcm...` </td>
      <td> </td>
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
- `examples`: Compile the `src/Examples` directory.
- `tools`: Compile the `src/Tools` directory.
- `tests`: Compile unit tests in the `src/UnitTests` directory (except tests for
  matrix formats, which have a separate target).
- `matrix-tests`: Compile unit tests for matrix formats.
- `doc`: Generate the documentation.

Additionally, `[options]` can be replaced with a list of options with the `--`
prefix that can be viewed by running `./install --help`.

Note that [CMake](https://cmake.org/) 3.24 or later is required when using the
`install` script.

## Usage   {#usage}

The following shows some of the most convenient ways to use TNL.

### Wrapper tnlcxx

`tnlcxx` is a wrapper which configures the build system (CMake) for simple situations where
the user needs to compile only one `.cpp` or `.cu` source file. The wrapper is available in a
separate [git repository](https://gitlab.com/tnl-project/tnlcxx).

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
The compiler flags used in the example project are explained in the following section.

### Important C++ compiler flags

- Enable the C++17 standard: `-std=c++17`
- Configure the include path: `-I /path/to/include`
    - If you installed TNL with the install script, the include path is
      `<prefix>/include`, where `<prefix>` is the installation path (it is
      `~/.local` by default).
    - If you want to include from the git repository directly, you need to
      specify `<git_repo>/src` as an include paths, where `<git_repo>` is the
      path where you have cloned the TNL git repository.
    - Instead of using the `-I` flag, you can set the `CPATH` environment
      variable to a colon-delimited list of include paths. Note that this may
      affect the build systems of other projects as well. For example:

          export CPATH="$HOME/.local/include:$CPATH"

- Enable optimizations: `-O3 -DNDEBUG` (you can also add
  `-march=native -mtune=native` to enable CPU-specific optimizations).
- Of course, there are many other useful compiler flags. For example, the
  flags that we use when developing TNL can be found in the
  [cxx_flags.cmake][cxx_flags.cmake] and [cuda_flags.cmake][cuda_flags.cmake]
  files in the Git repository.

[cxx_flags.cmake]: https://gitlab.com/tnl-project/tnl/-/blob/main/cmake/cxx_flags.cmake
[cuda_flags.cmake]: https://gitlab.com/tnl-project/tnl/-/blob/main/cmake/cuda_flags.cmake

### Compiler flags for parallel computing

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

### Environment variables   {#environment-variables}

If you installed some TNL tools or examples using the `install` script, we
recommend you to configure several environment variables for convenience. If you
used the default installation path `~/.local/`:

    export PATH="$PATH:$HOME/.local/bin"

These commands can be added to the initialization scripts of your favourite
shell, e.g. `.bash_profile`.
