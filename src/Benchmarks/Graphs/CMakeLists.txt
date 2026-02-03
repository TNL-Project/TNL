set(GRAPH_BENCHMARKS bfs sssp mst)

find_package(Boost OPTIONAL_COMPONENTS graph)
if(Boost_GRAPH_FOUND)
    message(STATUS "Boost found ${Boost_INCLUDE_DIRS}")
else()
    if(Boost_FOUND)
        message(
            STATUS "Boost found ${Boost_INCLUDE_DIRS} but not Boost::Graph. Benchmarks with Boost::Graph will not be built."
        )
    else()
        message(STATUS "Boost not found. Benchmarks with Boost::Graph will not be built.")
    endif()
endif()

if(TNL_BUILD_CUDA)
    # Find Gunrock
    #find_path(GUNROCK_INCLUDE_DIR gunrock/algorithms/bfs.hxx)
    if(GUNROCK_INCLUDE_DIR)
        message(STATUS "Gunrock found ${GUNROCK_INCLUDE_DIR}")
    else()
        message(STATUS "Gunrock not found.")
    endif()

    set(GRAPH_BENCHMARK_TARGETS_CUDA)

    foreach(bench IN LISTS GRAPH_BENCHMARKS)
        set(target tnl-benchmark-graphs-${bench}-cuda)
        add_executable(${target} tnl-benchmark-graphs-${bench}.cu)
        target_link_libraries(${target} PUBLIC TNL::TNL)

        if(NOT bench STREQUAL "mst")
            target_compile_options(${target} PUBLIC ${GUNROCK_FLAGS})
        endif()

        if(Boost_GRAPH_FOUND)
            target_link_libraries(${target} PUBLIC Boost::graph)
            target_compile_definitions(${target} PUBLIC "-DHAVE_BOOST")
        endif()

        if(GUNROCK_INCLUDE_DIR)
            target_compile_definitions(${target} PUBLIC HAVE_GUNROCK SM_TARGET=${CMAKE_CUDA_ARCHITECTURES})
            target_include_directories(${target} PUBLIC ${GUNROCK_INCLUDE_DIR})
        endif()

        list(APPEND GRAPH_BENCHMARK_TARGETS_CUDA ${target})
    endforeach()

    # NOTE: the following parameters are just for performance debugging
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
    # target_compile_options( tnl-benchmark-graphs PUBLIC --resource-usage )
    # set_target_properties( tnl-benchmark-graphs PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )

    install(TARGETS ${GRAPH_BENCHMARK_TARGETS_CUDA} RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

set(GRAPH_BENCHMARK_TARGETS)

foreach(bench IN LISTS GRAPH_BENCHMARKS)
    set(target tnl-benchmark-graphs-${bench})
    add_executable(${target} tnl-benchmark-graphs-${bench}.cpp)
    target_link_libraries(${target} PUBLIC TNL::TNL)

    if(Boost_GRAPH_FOUND)
        target_link_libraries(${target} PUBLIC Boost::graph)
        target_compile_definitions(${target} PUBLIC "-DHAVE_BOOST")
    endif()

    list(APPEND GRAPH_BENCHMARK_TARGETS ${target})
endforeach()

install(TARGETS ${GRAPH_BENCHMARK_TARGETS} RUNTIME DESTINATION bin COMPONENT benchmarks)
