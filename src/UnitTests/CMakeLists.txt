add_subdirectory(Algorithms)
add_subdirectory(Arithmetics)
add_subdirectory(Containers)
add_subdirectory(Graphs)
add_subdirectory(Matrices)
add_subdirectory(Meshes)
add_subdirectory(MPI)
add_subdirectory(Pointers)
add_subdirectory(Solvers)

set(CPP_TESTS
    AllocatorsTest
    AssertTest
    FileTest
    base64Test
    CustomScalarTest
    DiscreteMathTest
    FileNameTest
    MathTest
    ObjectTest
    StringTest
    TimerTest
    TypeInfoTest
    TypeTraitsTest
)
set(CUDA_TESTS AllocatorsTestCuda AssertTestCuda FileTestCuda)
set(HIP_TESTS AllocatorsTestHip AssertTestHip FileTestHip)

foreach(target IN ITEMS ${CPP_TESTS})
    add_executable(${target} ${target}.cpp)
    target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
    target_link_libraries(${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES})
    target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
    add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
endforeach()

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${CUDA_TESTS})
        add_executable(${target} ${target}.cu)
        target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()

if(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${HIP_TESTS})
        add_executable(${target} ${target}.hip)
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL_HIP ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()

# special test needing external library
find_package(ZLIB)
if(ZLIB_FOUND)
    foreach(target IN ITEMS zlibCompressionTest)
        add_executable(${target} ${target}.cpp)
        target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
        target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} ZLIB::ZLIB)
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()

if(TNL_BUILD_MPI)
    find_package(Hypre)

    if(HYPRE_FOUND)
        foreach(target IN ITEMS HypreTest)
            add_executable(${target} ${target}.cpp)
            target_link_libraries(${target} PUBLIC TNL::TNL_CXX)

            # enable MPI support in TNL
            target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
            # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
            target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

            target_compile_definitions(${target} PUBLIC "-DHAVE_HYPRE")
            target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
            target_include_directories(${target} SYSTEM PUBLIC ${HYPRE_INCLUDE_DIRS})
            target_link_libraries(${target} PUBLIC ${TESTS_LIBRARIES} ${HYPRE_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})

            add_test_mpi(NAME ${target} NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
            add_test_mpi(
                NAME ${target}_nodistr NPROC 1 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
        endforeach()
    endif()
endif()
