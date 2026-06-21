# Unit tests architecture

## Sources for several backends

A test which is built for several backends (host, CUDA, HIP) is written once,
typically in a header file `<name>.h` which defines the tests and includes
`main.h` (or `main_mpi.h`). Do not add `.cpp`, `.cu` and `.hip` files (or
symbolic links) which only include the header: the CMake function
`tnl_add_test_executables` generates them in the build directory.

### `tnl_add_test_executables`

```cmake
tnl_add_test_executables(<name>
    SOURCES <source>...
    [CUDA_SOURCES <source>...] [HIP_SOURCES <source>...]
    [BACKENDS <backend>...] [SINGLE] [OPTIONAL_HOST]
    [MPI [NPROC <nproc>]]
    [LABELS <label>...]
    [COMPILE_DEFINITIONS <definition>...]
    [LINK_LIBRARIES <library>...]
    [CUDA_LINK_LIBRARIES <library>...] [HIP_LINK_LIBRARIES <library>...])
```

The function adds one executable for each enabled backend, compiles and links
it with the flags for tests, and registers it with CTest (using the functions
described below for HIP and MPI tests). The executables and the tests are named
`<name>` (host), `<name>-cuda` and `<name>-hip`.

- `SOURCES` are included by the generated source file of each backend. When
  there is only one source with the extension of the backend (`.cpp`, `.cu`,
  `.hip`), it is compiled directly. `CUDA_SOURCES` and `HIP_SOURCES` replace
  `SOURCES` for one backend when the sources really differ between backends.
- `BACKENDS` limits the backends (`HOST`, `CUDA`, `HIP`), e.g. `BACKENDS HOST`
  for host-only tests.
- `SINGLE` builds only one executable named `<name>` for the first enabled
  backend in `BACKENDS`, e.g. `BACKENDS CUDA HIP HOST SINGLE` builds the test
  for the GPU if available, otherwise for the host.
- `OPTIONAL_HOST` builds the host executable only when `TNL_BUILD_CPP_TARGETS`
  is true, i.e. it is skipped in the CUDA and HIP CI jobs. Use it for tests
  built for several backends, so that the GPU jobs do not build the host
  variant as well.
- `MPI` enables MPI and registers the test with `NPROC` processes (default: 4)
  and with 1 process (`<target>_nodistr`).

Examples:

```cmake
# Vector test for all backends: VectorTest, VectorTest-cuda, VectorTest-hip
tnl_add_test_executables(VectorTest SOURCES VectorTest.h)

# header without main()
tnl_add_test_executables(VectorBinaryOperationsTest SOURCES VectorBinaryOperationsTest.h ../main.h)

# matrix test with a label
tnl_add_test_executables(DenseMatrixTest SOURCES DenseMatrixTest.h LABELS Matrices)
```

## CMake test registration

TNL provides four wrapper functions in `cmake/testing.cmake` for registering
tests with CTest. Always use these instead of bare `add_test()` —
`tnl_add_test_executables` uses them automatically. Test executables which are
registered manually must be linked to the `TNL_tests` interface target, which
provides the compiler and linker flags for tests and the GTest libraries.

### `tnl_add_test`

```cmake
tnl_add_test(<name> <command> [<arg>...])
tnl_add_test(NAME <name> COMMAND <command> [<arg>...])
```

Registers a plain test with CTest and bounds its OpenMP thread count by
setting the `ENVIRONMENT` property to
`OMP_NUM_THREADS=${TNL_TEST_OMP_NUM_THREADS}`. The cache variable
`TNL_TEST_OMP_NUM_THREADS` defaults to 4 and can be overridden at configure
time, e.g. with `-DTNL_TEST_OMP_NUM_THREADS=2`.

The bound is needed because CTest runs several tests in parallel (see the
`execution.jobs` option in `CMakePresets.json`). Without it, each test process
starts as many OpenMP threads as there are CPU cores, so the number of running
threads is the core count multiplied by the number of concurrent tests.

Tests which set their own `ENVIRONMENT` property keep it, because the function
does not overwrite an existing `ENVIRONMENT` property. This is how the MPI
tests keep `OMP_NUM_THREADS=1`. Tests added with `tnl_add_test_executables`
(host and CUDA) and with `tnl_add_hip_test` are bounded automatically; only
test executables which are registered by hand need an explicit call:

```cmake
tnl_add_test(
    NAME MultiPrecisionTest
    COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MultiPrecisionTest${CMAKE_EXECUTABLE_SUFFIX}"
)
```

A test registered with bare `add_test()` gets no `ENVIRONMENT` property and
runs with the OpenMP default, i.e. one thread per CPU core. The properties can
be inspected without running the tests:

```bash
ctest --show-only=json-v1 | grep OMP_NUM_THREADS
```

### `tnl_add_hip_test`

```cmake
tnl_add_hip_test(NAME <name> COMMAND <command> [<arg>...])
```

Registers a HIP test with `RESOURCE_LOCK gpu` and `LABELS HIP`. The lock
serializes execution of all HIP tests, preventing amdgpu hardware queue
oversubscription, which causes kernel warnings and potential GPU lockups on
AMD GPUs.

Use this in place of `add_test()` for all HIP test targets which are not
added with `tnl_add_test_executables`:

```cmake
if(TNL_BUILD_HIP)
    # ${target} is an existing executable target, e.g. created with
    # tnl_add_backend_executables or add_executable
    tnl_add_hip_test(NAME ${target} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
endif()
```

### `tnl_add_mpi_test`

```cmake
tnl_add_mpi_test(NAME <name> NPROC <nproc> COMMAND <command> [<arg>...])
```

Registers an MPI test that runs via `mpiexec` with `--oversubscribe`. Sets
`OMP_NUM_THREADS=1`, `LABELS MPI`, and `PROCESSORS <nproc>`.

### `tnl_add_hip_mpi_test`

```cmake
tnl_add_hip_mpi_test(NAME <name> NPROC <nproc> COMMAND <command> [<arg>...])
```

Registers a HIP+MPI test. Multi-process HIP+MPI tests cause deadlocks on a
single AMD GPU, so **only `NPROC=1` tests are registered** — calls with
`NPROC>1` are skipped with a `message(STATUS ...)` at configure time. Set
`RESOURCE_LOCK gpu`, `LABELS "HIP;MPI"`, `OMP_NUM_THREADS=1`, and
`PROCESSORS <nproc>`.

Call this for all HIP+MPI test registrations (including multi-process ones);
the function handles skipping automatically:

```cmake
if(TNL_BUILD_HIP)
    tnl_add_hip_mpi_test(
        NAME distributedScanTest-hip         # skipped: NPROC > 1
        NPROC 4
        COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest-hip${CMAKE_EXECUTABLE_SUFFIX}"
    )
    tnl_add_hip_mpi_test(
        NAME distributedScanTest-hip_nodistr # registered: NPROC == 1
        NPROC 1
        COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest-hip${CMAKE_EXECUTABLE_SUFFIX}"
    )
endif()
```

### Adding extra labels

When adding labels on top of those set by the registration functions, always
use `APPEND` to avoid overwriting existing labels:

```cmake
# correct: preserves HIP label from tnl_add_hip_test
set_property(TEST ${target} APPEND PROPERTY LABELS Matrices)

# wrong: overwrites HIP label with only Matrices
set_property(TEST ${target} PROPERTY LABELS Matrices)
```

### CTest presets

The `hip-tests` preset in `CMakePresets.json` runs all HIP-labeled tests
serialized:

```bash
ctest --preset hip-tests
```

## Typed-test lists

Test suites instantiated from `::testing::Types<...>` branch on the compiler:
the host branch (used in plain C++ builds) can keep the full Cartesian product
of real and index types, while the `__CUDACC__` and `__HIP__` branches need to
be reduced to a covering set.

The reason is compile time and memory usage of the compiler. Every device
instantiation emits its own copy of the same kernels and all instances within a
translation unit are compiled sequentially, which takes a lot of time and
memory. Compiling all tests is slow and frequently caused OOM errors on
memory-limited hosts.

A covering set keeps every axis — real type, index type, and other template
parameters — covered at least once, pairing the axes across entries instead of
exploding them into the full product:

```cpp
#elif defined( __CUDACC__ )
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::RowMajorEllpack< TNL::Devices::Cuda, int > >,
   TNL::Algorithms::Segments::SortedSegments< TNL::Algorithms::Segments::ColumnMajorEllpack< TNL::Devices::Cuda, long > >
```

Suites that remain expensive even as a covering set, such as the matrix tests,
may reduce the real-type axis further to a single floating-point and a single
integral type, plus a complex type where the host list has one. The index-type
axis and the remaining template parameters must still be covered.

Examples are the sorted segment test headers (`SegmentsTest_Sorted*.h` and
their `Reduce`/`Traverse` variants), the sparse matrix-vector product tests,
and the CSR sparse matrix reduce/traverse tests. Host lists may keep their
full type lists.

Mesh reader tests restrict the build-config tags similarly: they only read
files written from `Mesh<DefaultConfig<T>>`, so all resolver candidates except
the default real and index types are disabled (see the `MyConfigTag`
specializations in `VTUReaderTest.cpp` and the other reader tests).

### CUDA extended lambdas in typed tests

NVCC forbids extended `__host__ __device__` lambdas inside private or
protected member functions. Google Test's `TYPED_TEST` macro generates
`TestBody()` as a **private** method, so any `__cuda_callable__` lambda
written directly inside a `TYPED_TEST` body fails to compile with NVCC.

**Workaround**: extract the test body into a free function template and
have the `TYPED_TEST` call it:

```cpp
template< typename GraphType >
void
test_something_impl()
{
   // lambda is now in a free function, not a private method
   auto pred = [=] __cuda_callable__( typename GraphType::IndexType v )
   {
      return v < 3;
   };
   // ... assertions ...
}

TYPED_TEST( GraphTest, test_something )
{
   test_something_impl< typename TestFixture::GraphType >();
}
```

Rules for applying this pattern:

- Only extract tests that contain `__cuda_callable__` lambdas; tests
  without lambdas compile fine as-is.
- The free function is `template< typename GraphType >` and named
  `<test_name>_impl`, placed **before** the `TYPED_TEST`.
- Remove `using GraphType = typename TestFixture::GraphType;` inside
  the `_impl` function — `GraphType` is now the template parameter.
- Keep all other `using` declarations, comments, formatting, and
  assertions unchanged.
