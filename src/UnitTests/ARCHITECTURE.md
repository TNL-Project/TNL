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
