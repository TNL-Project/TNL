# Unit tests architecture

## CMake test registration

TNL provides three wrapper functions in `cmake/testing.cmake` for registering
tests with CTest. Always use these instead of bare `add_test()` for HIP and MPI
tests.

### `tnl_add_hip_test`

```cmake
tnl_add_hip_test(NAME <name> COMMAND <command> [<arg>...])
```

Registers a HIP test with `RUN_SERIAL TRUE` and `LABELS HIP`. Serialized
execution prevents amdgpu hardware queue oversubscription, which causes kernel
warnings and potential GPU lockups on AMD GPUs.

Use this in place of `add_test()` for all HIP test targets:

```cmake
if(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${HIP_TESTS})
        add_executable(${target} ${target}.hip)
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        tnl_add_hip_test(NAME ${target} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
    endforeach()
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
`RUN_SERIAL TRUE`, `LABELS "HIP;MPI"`, `OMP_NUM_THREADS=1`, and
`PROCESSORS <nproc>`.

Call this for all HIP+MPI test registrations (including multi-process ones);
the function handles skipping automatically:

```cmake
if(TNL_BUILD_HIP)
    tnl_add_hip_mpi_test(
        NAME distributedScanTestHip         # skipped: NPROC > 1
        NPROC 4
        COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestHip${CMAKE_EXECUTABLE_SUFFIX}"
    )
    tnl_add_hip_mpi_test(
        NAME distributedScanTestHip_nodistr # registered: NPROC == 1
        NPROC 1
        COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestHip${CMAKE_EXECUTABLE_SUFFIX}"
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

Examples are the sorted segment test headers (`SegmentsTest_Sorted*.h` and
their `Reduce`/`Traverse` variants), the sparse matrix-vector product tests,
and the CSR sparse matrix reduce/traverse tests. Host lists may keep their
full type lists.

Mesh reader tests restrict the build-config tags similarly: they only read
files written from `Mesh<DefaultConfig<T>>`, so all resolver candidates except
the default real and index types are disabled (see the `MyConfigTag`
specializations in `VTUReaderTest.cpp` and the other reader tests).
