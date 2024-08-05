set(COMMON_TESTS ODESolverTest_Euler ODESolverTest_Merson
                 StaticODESolverTest_Merson # This is to test the static ODE solver for CUDA and HIP
)

set(CPP_TESTS
    StaticODESolverTest_BogackiShampin
    StaticODESolverTest_CashKarp
    StaticODESolverTest_DormandPrince
    StaticODESolverTest_Euler
    StaticODESolverTest_Fehlberg2
    StaticODESolverTest_Fehlberg5
    StaticODESolverTest_Heun2
    StaticODESolverTest_Heun3
    StaticODESolverTest_Kutta
    StaticODESolverTest_Midpoint
    StaticODESolverTest_OriginalRungeKutta
    StaticODESolverTest_Ralston2
    StaticODESolverTest_Ralston3
    StaticODESolverTest_Ralston4
    StaticODESolverTest_Rule38
    StaticODESolverTest_SSPRK3
    StaticODESolverTest_VanDerHouwenWray
)

if(TNL_BUILD_CUDA)
    set(CUDA_TESTS ${CUDA_TESTS} ${COMMON_TESTS})
elseif(TNL_BUILD_HIP)
    set(HIP_TESTS ${HIP_TESTS} ${COMMON_TESTS})
else()
    set(CPP_TESTS ${CPP_TESTS} ${COMMON_TESTS})
endif()

foreach(target IN ITEMS ${CPP_TESTS})
    add_executable(${target} ${target}.cpp)
    target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
    target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
    target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
    add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
endforeach()

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${CUDA_TESTS})
        add_executable(${target} ${target}.cu)
        target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()

if(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${HIP_TESTS})
        add_executable(${target} ${target}.hip)
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    endforeach()
endif()
