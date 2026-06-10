# include CTest only when the project is top-level (not when it is added to
# the build tree of another project with add_subdirectory)
if(PROJECT_IS_TOP_LEVEL)
    # configure the project for testing with CTest/CDash
    include(CTest)

    if(TNL_USE_SYSTEM_GTEST OR TNL_OFFLINE_BUILD)
        # find GoogleTest installed in the local system
        find_package(GTest REQUIRED)
    else()
        # fetch and build GoogleTest from source
        include(FetchGoogleTest)
    endif()
    set(CXX_TESTS_FLAGS)
    set(CUDA_TESTS_FLAGS)
    set(HIP_TESTS_FLAGS)
    set(TESTS_LIBRARIES GTest::gtest_main)
    set(TESTS_LINKER_FLAGS "")

    if(TNL_BUILD_COVERAGE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # set compiler flags needed for code coverage
        set(CXX_TESTS_FLAGS ${CXX_TESTS_FLAGS} --coverage)
        set(CUDA_TESTS_FLAGS ${CUDA_TESTS_FLAGS} -Xcompiler --coverage)
        set(HIP_TESTS_FLAGS ${HIP_TESTS_FLAGS} --coverage)
        set(TESTS_LINKER_FLAGS ${TESTS_LINKER_FLAGS} --coverage)
    endif()
endif()

# function to simplify adding MPI tests
# usage: tnl_add_mpi_test(NAME <name> NPROC <nproc> COMMAND <command> [<arg>...])
function(tnl_add_mpi_test)
    # parse function arguments
    set(options)
    set(oneValueArgs NAME NPROC)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(ADD_TEST_MPI "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # set flags for MPIEXEC_EXECUTABLE
    set(mpiexec_flags "${MPIEXEC_NUMPROC_FLAG}" "${ADD_TEST_MPI_NPROC}" "--oversubscribe")

    add_test(NAME ${ADD_TEST_MPI_NAME} COMMAND "${MPIEXEC_EXECUTABLE}" ${mpiexec_flags} ${ADD_TEST_MPI_COMMAND})

    # set OMP_NUM_THREADS=1 to disable OpenMP for MPI tests
    # (NPROC may be even greater than the number of physical cores, so it would just slow down)
    set_property(TEST ${ADD_TEST_MPI_NAME} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    # add "MPI" label to the test
    set_property(TEST ${ADD_TEST_MPI_NAME} PROPERTY LABELS MPI)

    # set the number of processes used by the test
    set_property(TEST ${ADD_TEST_MPI_NAME} PROPERTY PROCESSORS ${ADD_TEST_MPI_NPROC})
endfunction()

# function to add a HIP test with RUN_SERIAL property
# Parallel execution of HIP tests oversubscribes the GPU hardware queues,
# causing amdgpu kernel warnings and potential lockups on AMD GPUs.
# usage: tnl_add_hip_test(NAME <name> COMMAND <command> [<arg>...])
function(tnl_add_hip_test)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(TNL_ADD_HIP_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_test(NAME ${TNL_ADD_HIP_TEST_NAME} COMMAND ${TNL_ADD_HIP_TEST_COMMAND})

    # serialize HIP tests to avoid GPU queue oversubscription
    set_property(TEST ${TNL_ADD_HIP_TEST_NAME} PROPERTY RUN_SERIAL TRUE)

    # add "HIP" label to the test for ctest filtering
    set_property(TEST ${TNL_ADD_HIP_TEST_NAME} PROPERTY LABELS HIP)
endfunction()

# function to add a HIP+MPI test
# Multi-process HIP+MPI tests cause deadlocks on a single AMD GPU, so only NPROC=1
# tests are registered. Tests with NPROC>1 are skipped with a status message.
# usage: tnl_add_hip_mpi_test(NAME <name> NPROC <nproc> COMMAND <command> [<arg>...])
function(tnl_add_hip_mpi_test)
    set(options)
    set(oneValueArgs NAME NPROC)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(TNL_ADD_HIP_MPI_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT TNL_ADD_HIP_MPI_TEST_NPROC EQUAL 1)
        message(
            STATUS
            "Skipping HIP+MPI test ${TNL_ADD_HIP_MPI_TEST_NAME} with NPROC=${TNL_ADD_HIP_MPI_TEST_NPROC} (deadlocks on a single AMD GPU)"
        )
        return()
    endif()

    # set flags for MPIEXEC_EXECUTABLE
    set(mpiexec_flags "${MPIEXEC_NUMPROC_FLAG}" "${TNL_ADD_HIP_MPI_TEST_NPROC}" "--oversubscribe")

    add_test(NAME ${TNL_ADD_HIP_MPI_TEST_NAME} COMMAND "${MPIEXEC_EXECUTABLE}" ${mpiexec_flags} ${TNL_ADD_HIP_MPI_TEST_COMMAND})

    # set OMP_NUM_THREADS=1 to disable OpenMP for MPI tests
    set_property(TEST ${TNL_ADD_HIP_MPI_TEST_NAME} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

    # serialize HIP tests to avoid GPU queue oversubscription
    set_property(TEST ${TNL_ADD_HIP_MPI_TEST_NAME} PROPERTY RUN_SERIAL TRUE)

    # add labels for filtering
    set_property(TEST ${TNL_ADD_HIP_MPI_TEST_NAME} PROPERTY LABELS "HIP;MPI")

    # set the number of processes used by the test
    set_property(TEST ${TNL_ADD_HIP_MPI_TEST_NAME} PROPERTY PROCESSORS ${TNL_ADD_HIP_MPI_TEST_NPROC})
endfunction()
