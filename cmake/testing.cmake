# interface target with the libraries and the compiler and linker flags for tests;
# every test executable links this target, so that all tests are built in the same way
add_library(TNL_tests INTERFACE)

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
    target_link_libraries(TNL_tests INTERFACE GTest::gtest_main)

    if(TNL_BUILD_COVERAGE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # set compiler and linker flags needed for code coverage
        target_compile_options(
            TNL_tests
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:--coverage>
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;--coverage>"
                $<$<COMPILE_LANGUAGE:HIP>:--coverage>
        )
        target_link_options(TNL_tests INTERFACE --coverage)
    endif()
endif()

# number of OpenMP threads used by a single unit test
# Tests are executed in parallel via `ctest -j`, so the per-test thread count is
# kept small to avoid oversubscription of the CPU cores. Without this, every test
# uses as many threads as there are cores (the OpenMP default) whenever
# OMP_NUM_THREADS is not set in the environment of the ctest process.
set(TNL_TEST_OMP_NUM_THREADS 4 CACHE STRING "Number of OpenMP threads used by unit tests")

# function to add a plain unit test with a bounded number of OpenMP threads
# usage: tnl_add_test(<name> <command> [<arg>...]) or tnl_add_test(NAME <name> COMMAND <command> [<arg>...])
function(tnl_add_test)
    add_test(${ARGN})

    if(ARGV0 STREQUAL "NAME")
        set(test_name ${ARGV1})
    else()
        set(test_name ${ARGV0})
    endif()

    # keep the environment of tests that configure it themselves (e.g. MPI tests)
    get_test_property(${test_name} ENVIRONMENT test_environment)
    if(NOT test_environment)
        set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${TNL_TEST_OMP_NUM_THREADS}")
    endif()
endfunction()

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

# function to add a HIP test that holds the GPU resource lock
# Parallel execution of GPU tests oversubscribes the GPU hardware queues,
# causing amdgpu kernel warnings and potential lockups on AMD GPUs, so all
# tests that use a GPU take the same resource lock and never run concurrently.
# usage: tnl_add_hip_test(NAME <name> COMMAND <command> [<arg>...])
function(tnl_add_hip_test)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(TNL_ADD_HIP_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # also bounds the number of OpenMP threads used by the test
    tnl_add_test(NAME ${TNL_ADD_HIP_TEST_NAME} COMMAND ${TNL_ADD_HIP_TEST_COMMAND})

    # serialize GPU tests to avoid oversubscription of the GPU hardware queues
    set_property(TEST ${TNL_ADD_HIP_TEST_NAME} PROPERTY RESOURCE_LOCK gpu)

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

    # serialize GPU tests to avoid oversubscription of the GPU hardware queues
    set_property(TEST ${TNL_ADD_HIP_MPI_TEST_NAME} PROPERTY RESOURCE_LOCK gpu)

    # add labels for filtering
    set_property(TEST ${TNL_ADD_HIP_MPI_TEST_NAME} PROPERTY LABELS "HIP;MPI")

    # set the number of processes used by the test
    set_property(TEST ${TNL_ADD_HIP_MPI_TEST_NAME} PROPERTY PROCESSORS ${TNL_ADD_HIP_MPI_TEST_NPROC})
endfunction()

# function to add test executables built from the same sources for several backends
# usage:
#   tnl_add_test_executables(<name>
#       SOURCES <source>...
#       [CUDA_SOURCES <source>...] [HIP_SOURCES <source>...]
#       [BACKENDS <backend>...] [SINGLE] [OPTIONAL_HOST]
#       [MPI [NPROC <nproc>]]
#       [LABELS <label>...]
#       [COMPILE_DEFINITIONS <definition>...]
#       [LINK_LIBRARIES <library>...]
#       [CUDA_LINK_LIBRARIES <library>...] [HIP_LINK_LIBRARIES <library>...])
# - adds the executables with tnl_add_backend_executables (see target_helpers.cmake),
#   which is the non-test variant of this function
# - the executables are linked to the TNL_tests target, which provides the
#   compiler and linker flags for tests and the GTest libraries
# - the executables are registered as tests in CTest and named <name> (host),
#   <name>-cuda and <name>-hip (or just <name> with SINGLE)
# - MPI enables MPI support and registers each test with NPROC processes
#   (default: 4) and with 1 process (<target>_nodistr)
# - LABELS are added to all registered tests
function(tnl_add_test_executables name)
    set(options MPI SINGLE OPTIONAL_HOST)
    set(oneValueArgs NPROC)
    set(multiValueArgs
        LABELS
        SOURCES
        CUDA_SOURCES
        HIP_SOURCES
        BACKENDS
        COMPILE_DEFINITIONS
        LINK_LIBRARIES
        CUDA_LINK_LIBRARIES
        HIP_LINK_LIBRARIES
    )
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "tnl_add_test_executables: unknown arguments for '${name}': ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(ARG_MPI AND NOT TNL_BUILD_MPI)
        return()
    endif()

    # forward the arguments of tnl_add_backend_executables
    set(forward)
    foreach(option IN ITEMS SINGLE OPTIONAL_HOST)
        if(ARG_${option})
            list(APPEND forward ${option})
        endif()
    endforeach()
    foreach(
        arg
        IN
        ITEMS SOURCES CUDA_SOURCES HIP_SOURCES BACKENDS CUDA_LINK_LIBRARIES HIP_LINK_LIBRARIES
    )
        if(ARG_${arg})
            list(APPEND forward ${arg} ${ARG_${arg}})
        endif()
    endforeach()

    if(NOT ARG_NPROC)
        set(ARG_NPROC 4)
    endif()

    set(definitions ${ARG_COMPILE_DEFINITIONS})
    set(libraries TNL_tests ${ARG_LINK_LIBRARIES})
    if(ARG_MPI)
        # enable MPI support in TNL
        list(APPEND definitions HAVE_MPI)
        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        list(APPEND libraries MPI::MPI_CXX)
    endif()

    tnl_add_backend_executables(
        ${name}
        ${forward}
        COMPILE_DEFINITIONS ${definitions}
        LINK_LIBRARIES ${libraries}
        OUTPUT_TARGETS targets
        OUTPUT_BACKENDS backends
    )

    foreach(target backend IN ZIP_LISTS targets backends)
        set(command "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
        if(ARG_MPI AND backend STREQUAL "HIP")
            tnl_add_hip_mpi_test(NAME ${target} NPROC ${ARG_NPROC} COMMAND "${command}")
            tnl_add_hip_mpi_test(NAME ${target}_nodistr NPROC 1 COMMAND "${command}")
        elseif(ARG_MPI)
            tnl_add_mpi_test(NAME ${target} NPROC ${ARG_NPROC} COMMAND "${command}")
            tnl_add_mpi_test(NAME ${target}_nodistr NPROC 1 COMMAND "${command}")
        elseif(backend STREQUAL "HIP")
            tnl_add_hip_test(NAME ${target} COMMAND "${command}")
        else()
            tnl_add_test(NAME ${target} COMMAND "${command}")
        endif()

        if(ARG_LABELS)
            foreach(test IN ITEMS ${target} ${target}_nodistr)
                if(TEST ${test})
                    set_property(TEST ${test} APPEND PROPERTY LABELS ${ARG_LABELS})
                endif()
            endforeach()
        endif()
    endforeach()
endfunction()

# function to add the tests of a type-parameterized test suite for several formats
# (e.g. formats of segments)
# usage:
#   tnl_add_format_tests(<name>
#       SUITE <suite> PREFIX <prefix> TYPES <types>
#       HEADERS <header>... FORMATS <format>...
#       [FORMAT_TYPES <format>:<types>...]
#       [EXTRA <format>:<header>...]
#       [LABELS <label>...])
# - for each format, the header <build_dir>/generated/<name>_<format>.h is generated;
#   it includes the HEADERS and the EXTRA headers of the format, instantiates the
#   test suite SUITE for the type list TYPES with the PREFIX and includes the main
#   function of the unit tests
# - the placeholder <FORMAT> in TYPES and PREFIX is replaced by the name of the
#   format and FORMAT_TYPES replaces TYPES for some formats
# - the test executables are added with tnl_add_test_executables from the
#   generated header, so they are named <name>_<format> (host),
#   <name>_<format>-cuda and <name>_<format>-hip
# - LABELS are added to all registered tests
function(tnl_add_format_tests name)
    set(options)
    set(oneValueArgs SUITE PREFIX TYPES)
    set(multiValueArgs
        HEADERS
        FORMATS
        FORMAT_TYPES
        EXTRA
        LABELS
    )
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "tnl_add_format_tests: unknown arguments for '${name}': ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    foreach(
        arg
        IN
        ITEMS SUITE PREFIX TYPES HEADERS FORMATS
    )
        if(NOT ARG_${arg})
            message(FATAL_ERROR "tnl_add_format_tests: ${arg} is not set for '${name}'")
        endif()
    endforeach()

    set(labels)
    if(ARG_LABELS)
        set(labels LABELS ${ARG_LABELS})
    endif()

    foreach(format IN LISTS ARG_FORMATS)
        # resolve the type list, the prefix and the extra headers of the format
        string(REPLACE "<FORMAT>" "${format}" types "${ARG_TYPES}")
        foreach(item IN LISTS ARG_FORMAT_TYPES)
            if(item MATCHES "^${format}:(.+)$")
                set(types "${CMAKE_MATCH_1}")
            endif()
        endforeach()
        string(REPLACE "<FORMAT>" "${format}" prefix "${ARG_PREFIX}")
        set(headers ${ARG_HEADERS})
        foreach(item IN LISTS ARG_EXTRA)
            if(item MATCHES "^${format}:(.+)$")
                list(APPEND headers "${CMAKE_MATCH_1}")
            endif()
        endforeach()

        set(content "// Generated by tnl_add_format_tests in ${CMAKE_CURRENT_LIST_FILE}, do not edit.\n\n")
        foreach(header IN LISTS headers)
            get_filename_component(header "${header}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
            if(NOT EXISTS "${header}")
                message(FATAL_ERROR "tnl_add_format_tests: header '${header}' of '${name}_${format}' does not exist")
            endif()
            string(APPEND content "#include \"${header}\"\n")
        endforeach()
        string(
            APPEND content
            "\nusing InstantiatedTestTypes = ${types};\n"
            "INSTANTIATE_TYPED_TEST_SUITE_P( ${prefix}, ${ARG_SUITE}, InstantiatedTestTypes );\n\n"
            "#include \"${PROJECT_SOURCE_DIR}/src/UnitTests/main.h\"\n"
        )

        set(header "${CMAKE_CURRENT_BINARY_DIR}/generated/${name}_${format}.h")
        file(CONFIGURE OUTPUT "${header}" CONTENT "${content}" @ONLY)

        tnl_add_test_executables(${name}_${format} SOURCES "${header}" ${labels})
    endforeach()
endfunction()
