add_subdirectory(Algorithms)
add_subdirectory(Arithmetics)
add_subdirectory(Containers)
add_subdirectory(Graphs)
add_subdirectory(Matrices)
add_subdirectory(Meshes)
add_subdirectory(MPI)
add_subdirectory(Pointers)
add_subdirectory(Solvers)

set(HOST_TESTS
    base64Test
    BenchmarksTest
    CustomScalarTest
    DiscreteMathTest
    MathTest
    ObjectTest
    StringTest
    TimerTest
    TypeInfoTest
    TypeTraitsTest
)

foreach(test IN ITEMS ${HOST_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.cpp BACKENDS HOST)
endforeach()

tnl_add_test_executables(LaunchConfigurationTest SOURCES LaunchConfigurationTest.h main.h)

# tests with different sources for each backend
foreach(test IN ITEMS AllocatorsTest AssertTest FileTest)
    tnl_add_test_executables(${test} SOURCES ${test}.cpp CUDA_SOURCES ${test}Cuda.cu HIP_SOURCES ${test}Hip.hip)
endforeach()

# special test needing external library
find_package(ZLIB)
if(ZLIB_FOUND)
    tnl_add_test_executables(
        zlibCompressionTest
        SOURCES zlibCompressionTest.cpp
        BACKENDS HOST
        COMPILE_DEFINITIONS HAVE_ZLIB
        LINK_LIBRARIES ZLIB::ZLIB
    )
endif()

if(TNL_BUILD_MPI)
    find_package(Hypre)

    if(HYPRE_FOUND)
        foreach(target IN ITEMS HypreTest)
            add_executable(${target} ${target}.cpp)
            target_link_libraries(${target} PUBLIC TNL::TNL)

            # enable MPI support in TNL
            target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
            # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
            target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

            target_compile_definitions(${target} PUBLIC "-DHAVE_HYPRE")
            target_include_directories(${target} SYSTEM PUBLIC ${HYPRE_INCLUDE_DIRS})
            target_link_libraries(${target} PUBLIC ${HYPRE_LIBRARIES} TNL_tests)

            tnl_add_mpi_test(
                NAME ${target}
                NPROC 4
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
            tnl_add_mpi_test(
                NAME ${target}_nodistr
                NPROC 1
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
        endforeach()
    endif()
endif()
