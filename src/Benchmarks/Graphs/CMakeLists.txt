set(GRAPH_BENCHMARKS bfs cc coloring scc sssp mst)

find_package(Boost OPTIONAL_COMPONENTS graph)
if(Boost_GRAPH_FOUND)
    message(STATUS "Boost found ${Boost_INCLUDE_DIRS}")
else()
    if(Boost_FOUND)
        message(
            STATUS
            "Boost found ${Boost_INCLUDE_DIRS} but not Boost::Graph. Benchmarks with Boost::Graph will not be built."
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

        file(
            WRITE ${CMAKE_CURRENT_BINARY_DIR}/gunrock-test.cpp
            "#include <gunrock/algorithms/bfs.hxx>
            #include <gunrock/algorithms/sssp.hxx>
            int main() { return 0; }
            "
        )

        try_compile(
            GUNROCK_TEST
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/gunrock-test.cpp
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${GUNROCK_INCLUDE_DIR}"
            OUTPUT_VARIABLE GUNROCK_TEST_OUTPUT
        )
        if(GUNROCK_COMPILED)
            message(STATUS "Gunrock compiled successfully.")
            set(GUNROCK_FLAGS -DHAVE_GUNROCK ${Gunrock_INCLUDE_DIR})
        else()
            message(STATUS "Gunrock failed to compile, will be omitted in the benchmark.")
        endif()
    else()
        message(STATUS "Gunrock not found.")
    endif()
endif()

set(GRAPH_BENCHMARK_TARGETS)

foreach(bench IN LISTS GRAPH_BENCHMARKS)
    tnl_add_backend_executables(
        tnl-benchmark-graphs-${bench}
        SOURCES tnl-benchmark-graphs-${bench}.h
        BACKENDS HOST CUDA HIP
        OUTPUT_TARGETS targets
        OUTPUT_BACKENDS backends
    )

    if(Boost_GRAPH_FOUND)
        foreach(target IN LISTS targets)
            target_link_libraries(${target} PUBLIC Boost::graph)
            target_compile_definitions(${target} PUBLIC "-DHAVE_BOOST")
        endforeach()
    endif()

    # options and libraries for the CUDA variants
    foreach(target backend IN ZIP_LISTS targets backends)
        if(NOT backend STREQUAL "CUDA")
            continue()
        endif()
        if(NOT bench STREQUAL "mst" AND GUNROCK_FLAGS)
            target_compile_options(${target} PUBLIC ${GUNROCK_FLAGS})
        endif()
        if(GUNROCK_INCLUDE_DIR)
            target_compile_definitions(${target} PUBLIC HAVE_GUNROCK SM_TARGET=${CMAKE_CUDA_ARCHITECTURES})
            target_include_directories(${target} PUBLIC ${GUNROCK_INCLUDE_DIR})
        endif()
    endforeach()

    list(APPEND GRAPH_BENCHMARK_TARGETS ${targets})
endforeach()

# NOTE: the following parameters are just for performance debugging
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
# target_compile_options( tnl-benchmark-graphs-bfs-cuda PUBLIC --resource-usage )
# set_target_properties( tnl-benchmark-graphs-bfs-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )

if(GRAPH_BENCHMARK_TARGETS)
    install(TARGETS ${GRAPH_BENCHMARK_TARGETS} RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
