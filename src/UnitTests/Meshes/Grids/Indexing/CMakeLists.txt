set(CPP_TESTS Grid1DTest_Indexing Grid2DTest_Indexing Grid3DTest_Indexing)

foreach(target IN ITEMS ${CPP_TESTS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PUBLIC TNL::TNL TNL_tests)
    tnl_add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
endforeach()
