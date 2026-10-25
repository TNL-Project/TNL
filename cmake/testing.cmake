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

# function to get the sources of a test, which may be split into several translation
# units compiled separately and linked into one executable to reduce the compile time and
# memory of each compiler process. SPLIT_TESTS is a list of items <name>:<parts>. When the
# test NAME is listed there, the source is replaced by <parts> generated sources, each of
# which defines TNL_TEST_PART and TNL_TEST_PARTS and includes the original source. The
# source must instantiate the tests with TNL_INSTANTIATE_TYPED_TEST_SUITE_P (see
# src/UnitTests/TypedTestSplit.h) and the number of parts must not exceed the number of
# types in the instantiation. The split is disabled when TNL_SPLIT_UNIT_TESTS is OFF.
# usage: tnl_test_sources(<output_variable> NAME <name> SOURCE <source> [SPLIT_TESTS <name>:<parts>...])
function(tnl_test_sources output_variable)
    set(options)
    set(oneValueArgs NAME SOURCE)
    set(multiValueArgs SPLIT_TESTS)
    cmake_parse_arguments(TNL_TEST_SOURCES "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(parts)
    if(NOT TNL_SPLIT_UNIT_TESTS)
        set(TNL_TEST_SOURCES_SPLIT_TESTS)
    endif()
    foreach(item IN LISTS TNL_TEST_SOURCES_SPLIT_TESTS)
        if(item MATCHES "^${TNL_TEST_SOURCES_NAME}:([0-9]+)$")
            set(parts ${CMAKE_MATCH_1})
        endif()
    endforeach()
    if(NOT parts)
        set(${output_variable} ${TNL_TEST_SOURCES_SOURCE} PARENT_SCOPE)
        return()
    endif()

    get_filename_component(source "${TNL_TEST_SOURCES_SOURCE}" ABSOLUTE)
    get_filename_component(name "${source}" NAME_WE)
    get_filename_component(extension "${source}" LAST_EXT)

    set(sources)
    math(EXPR last_part "${parts} - 1")
    foreach(part RANGE ${last_part})
        set(part_source "${CMAKE_CURRENT_BINARY_DIR}/${name}_part${part}${extension}")
        file(
            CONFIGURE
            OUTPUT "${part_source}"
            CONTENT "#define TNL_TEST_PART ${part}\n#define TNL_TEST_PARTS ${parts}\n#include \"${source}\"\n"
        )
        list(APPEND sources "${part_source}")
    endforeach()

    set(${output_variable} ${sources} PARENT_SCOPE)
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

# function to add the tests of a type-parameterized test suite for several formats
# (e.g. formats of segments). For each format and each enabled backend, a source file is
# generated in the build directory. It includes the HEADERS and the EXTRA header of the
# format (if any), instantiates the test suite SUITE for the type list TYPES with the
# PREFIX, and defines main(). The placeholder <FORMAT> in TYPES and PREFIX is replaced by
# the name of the format and FORMAT_TYPES overrides TYPES for some formats. The test
# executables are named <name>_<format>, <name>_<format>_cuda and <name>_<format>_hip.
# usage: tnl_add_format_tests(NAME <name> SUITE <suite> PREFIX <prefix> TYPES <types>
#                             HEADERS <header>... FORMATS <format>...
#                             [FORMAT_TYPES <format>:<types>...] [EXTRA <format>:<header>...]
#                             [LABELS <label>...])
function(tnl_add_format_tests)
    set(options)
    set(oneValueArgs NAME SUITE PREFIX TYPES)
    set(multiValueArgs
        HEADERS
        FORMATS
        FORMAT_TYPES
        EXTRA
        LABELS
    )
    cmake_parse_arguments(TNL_FORMAT_TESTS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(backends cpp)
    if(TNL_BUILD_CUDA)
        list(APPEND backends cu)
    endif()
    if(TNL_BUILD_HIP)
        list(APPEND backends hip)
    endif()

    foreach(format IN LISTS TNL_FORMAT_TESTS_FORMATS)
        # resolve the type list, the prefix and the extra header of the format
        string(REPLACE "<FORMAT>" "${format}" types "${TNL_FORMAT_TESTS_TYPES}")
        foreach(item IN LISTS TNL_FORMAT_TESTS_FORMAT_TYPES)
            if(item MATCHES "^${format}:(.+)$")
                set(types "${CMAKE_MATCH_1}")
            endif()
        endforeach()
        string(REPLACE "<FORMAT>" "${format}" prefix "${TNL_FORMAT_TESTS_PREFIX}")
        set(headers ${TNL_FORMAT_TESTS_HEADERS})
        foreach(item IN LISTS TNL_FORMAT_TESTS_EXTRA)
            if(item MATCHES "^${format}:(.+)$")
                list(APPEND headers "${CMAKE_MATCH_1}")
            endif()
        endforeach()

        set(content "// Generated by tnl_add_format_tests in ${CMAKE_CURRENT_LIST_FILE}\n")
        foreach(header IN LISTS headers)
            get_filename_component(header "${header}" ABSOLUTE)
            string(APPEND content "#include \"${header}\"\n")
        endforeach()
        string(
            APPEND content
            "\nusing InstantiatedTestTypes = ${types};\n"
            "INSTANTIATE_TYPED_TEST_SUITE_P( ${prefix}, ${TNL_FORMAT_TESTS_SUITE}, InstantiatedTestTypes );\n\n"
            "#include \"${PROJECT_SOURCE_DIR}/src/UnitTests/main.h\"\n"
        )

        foreach(backend IN LISTS backends)
            if(backend STREQUAL "cu")
                set(target ${TNL_FORMAT_TESTS_NAME}_${format}_cuda)
            elseif(backend STREQUAL "hip")
                set(target ${TNL_FORMAT_TESTS_NAME}_${format}_hip)
            else()
                set(target ${TNL_FORMAT_TESTS_NAME}_${format})
            endif()
            set(source "${CMAKE_CURRENT_BINARY_DIR}/${target}.${backend}")
            file(CONFIGURE OUTPUT "${source}" CONTENT "${content}")

            add_executable(${target} ${source})
            target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
            set(command "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
            if(backend STREQUAL "cu")
                target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
                add_test(${target} ${command})
            elseif(backend STREQUAL "hip")
                target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
                tnl_add_hip_test(NAME ${target} COMMAND "${command}")
            else()
                target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
                add_test(${target} ${command})
            endif()
            if(TNL_FORMAT_TESTS_LABELS)
                set_property(TEST ${target} APPEND PROPERTY LABELS ${TNL_FORMAT_TESTS_LABELS})
            endif()
        endforeach()
    endforeach()
endfunction()
