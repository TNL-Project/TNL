set(COMMON_TESTS
    Grid1DTest_Neighbours_0D
    Grid1DTest_Neighbours_1D
    Grid2DTest_Neighbours_0D
    Grid2DTest_Neighbours_1D
    Grid2DTest_Neighbours_2D
    Grid3DTest_Neighbours_0D
    Grid3DTest_Neighbours_1D
    Grid3DTest_Neighbours_2D
    Grid3DTest_Neighbours_3D
)

set(CPP_TESTS ${COMMON_TESTS})
set(CUDA_TESTS ${COMMON_TESTS})
set(HIP_TESTS ${COMMON_TESTS})

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${CUDA_TESTS})
        add_executable(${target} ${target}.cu)
        target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
elseif(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${HIP_TESTS})
        add_executable(${target} ${target}.hip)
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
else()
    foreach(target IN ITEMS ${CPP_TESTS})
        add_executable(${target} ${target}.cpp)
        target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()
