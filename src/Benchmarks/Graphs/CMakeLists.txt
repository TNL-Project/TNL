find_package(Boost OPTIONAL_COMPONENTS graph)
if(Boost_GRAPH_FOUND)
    message(STATUS "Boost found ${Boost_INCLUDE_DIRS}")
else()
    if(Boost_FOUND)
        message(STATUS "Boost found ${Boost_INCLUDE_DIRS} but not Boost::Graph. Benchmarks with Boost::Graph will not be built.")
    else()
        message(STATUS "Boost not found. Benchmarks with Boost::Graph will not be built.")
    endif()
endif()

if(TNL_BUILD_CUDA)
    # Find Gunrock
    find_path(GUNROCK_INCLUDE_DIR gunrock/algorithms/bfs.hxx)
    if(GUNROCK_INCLUDE_DIR)
        message(STATUS "Gunrock found ${GUNROCK_INCLUDE_DIR}")
        set(GUNROCK_FLAGS -DHAVE_GUNROCK ${Gunrock_INCLUDE_DIR} -D SM_TARGET=${CMAKE_CUDA_ARCHITECTURES})
    else()
        message(STATUS "Gunrock not found.")
    endif()

    add_executable(tnl-benchmark-graphs-cuda tnl-benchmark-graphs.cu)
    target_compile_options(tnl-benchmark-graphs-cuda PUBLIC ${GUNROCK_FLAGS})
    target_link_libraries(tnl-benchmark-graphs-cuda PUBLIC TNL::TNL)

    if(Boost_GRAPH_FOUND)
        target_link_libraries(tnl-benchmark-graphs-cuda PUBLIC Boost::graph)
        target_compile_definitions(tnl-benchmark-graphs-cuda PUBLIC "-DHAVE_BOOST")
    endif()

    # NOTE: the following parameters are just for performance debugging
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
    # target_compile_options( tnl-benchmark-graphs PUBLIC --resource-usage )
    # set_target_properties( tnl-benchmark-graphs PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )

    install(TARGETS tnl-benchmark-graphs-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

add_executable(tnl-benchmark-graphs tnl-benchmark-graphs.cpp)
target_link_libraries(tnl-benchmark-graphs PUBLIC TNL::TNL)

if(Boost_GRAPH_FOUND)
    target_link_libraries(tnl-benchmark-graphs PUBLIC Boost::graph)
    target_compile_definitions(tnl-benchmark-graphs PUBLIC "-DHAVE_BOOST")
endif()

install(TARGETS tnl-benchmark-graphs RUNTIME DESTINATION bin COMPONENT benchmarks)
