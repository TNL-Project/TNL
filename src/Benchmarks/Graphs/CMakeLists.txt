set(GRAPH_BENCHMARKS
    bfs
    cc
    coloring
    scc
    sssp
    mst
)

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
    # Find Gunrock (header-only CUDA library, requires nvcc to compile)
    find_path(
        GUNROCK_INCLUDE_DIR
        NAMES gunrock/algorithms/bfs.hxx
        PATHS ${GUNROCK_ROOT} $ENV{GUNROCK_ROOT} $ENV{HOME}/.local/include /usr/local/include
    )
    if(GUNROCK_INCLUDE_DIR)
        message(STATUS "Gunrock found: ${GUNROCK_INCLUDE_DIR}")
        set(GUNROCK_FOUND TRUE)
    else()
        message(STATUS "Gunrock not found. Benchmarks with Gunrock will be omitted.")
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
        if(GUNROCK_FOUND AND NOT bench STREQUAL "mst")
            target_compile_definitions(${target} PUBLIC HAVE_GUNROCK SM_TARGET=86 HIP_BACKEND=1 __HIP_PLATFORM_NVIDIA__=1)
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
