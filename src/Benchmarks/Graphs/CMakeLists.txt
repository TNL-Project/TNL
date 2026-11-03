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
    # SM_TARGET is the compute capability of the first CUDA architecture (e.g. 86 from "86-real"),
    # Gunrock treats 0 as unknown
    list(GET CMAKE_CUDA_ARCHITECTURES 0 GUNROCK_SM_TARGET)
    string(REGEX MATCH "^[0-9]+" GUNROCK_SM_TARGET "${GUNROCK_SM_TARGET}")
    if(NOT GUNROCK_SM_TARGET)
        set(GUNROCK_SM_TARGET 0)
    endif()
    set(GUNROCK_DEFINITIONS HAVE_GUNROCK SM_TARGET=${GUNROCK_SM_TARGET} HIP_BACKEND=1 __HIP_PLATFORM_NVIDIA__=1)
    if(GUNROCK_INCLUDE_DIR)
        message(STATUS "Gunrock found: ${GUNROCK_INCLUDE_DIR}")
        # GunrockBenchmark.h uses the Gunrock 2.x API (options_t, graph::build from csr_t),
        # older Gunrock versions are skipped instead of breaking the build.
        include(CheckSourceCompiles)
        list(TRANSFORM GUNROCK_DEFINITIONS PREPEND "-D" OUTPUT_VARIABLE CMAKE_REQUIRED_DEFINITIONS)
        set(CMAKE_REQUIRED_INCLUDES ${GUNROCK_INCLUDE_DIR})
        set(CMAKE_REQUIRED_FLAGS "-std=c++17 --extended-lambda --expt-relaxed-constexpr")
        check_source_compiles(
            CUDA
            "#include <gunrock/algorithms/bfs.hxx>
            #include <gunrock/algorithms/sssp.hxx>
            #include <gunrock/graph/build.hxx>
            using csr_t = gunrock::format::csr_t< gunrock::memory_space_t::device, int, int, float >;
            using graph_t = decltype( gunrock::graph::build< gunrock::memory_space_t::device, int, int, float >(
                std::declval< gunrock::graph::graph_properties_t >(), std::declval< csr_t& >() ) );
            int main() { gunrock::options_t options; options.enable_filter = false; return 0; }"
            GUNROCK_API_COMPATIBLE
        )
        unset(CMAKE_REQUIRED_DEFINITIONS)
        unset(CMAKE_REQUIRED_INCLUDES)
        unset(CMAKE_REQUIRED_FLAGS)
        if(GUNROCK_API_COMPATIBLE)
            set(GUNROCK_FOUND TRUE)
        else()
            message(
                STATUS
                "Gunrock in ${GUNROCK_INCLUDE_DIR} does not provide the 2.x API. Benchmarks with Gunrock will be omitted."
            )
        endif()
    else()
        message(STATUS "Gunrock not found. Benchmarks with Gunrock will be omitted.")
    endif()

    set(GRAPH_BENCHMARK_TARGETS_CUDA)

    foreach(bench IN LISTS GRAPH_BENCHMARKS)
        set(target tnl-benchmark-graphs-${bench}-cuda)
        add_executable(${target} tnl-benchmark-graphs-${bench}.cu)
        target_link_libraries(${target} PUBLIC TNL::TNL)

        if(Boost_GRAPH_FOUND)
            target_link_libraries(${target} PUBLIC Boost::graph)
            target_compile_definitions(${target} PUBLIC "-DHAVE_BOOST")
        endif()

        if(GUNROCK_FOUND AND NOT bench STREQUAL "mst")
            target_compile_definitions(${target} PUBLIC ${GUNROCK_DEFINITIONS})
            target_include_directories(${target} SYSTEM PUBLIC ${GUNROCK_INCLUDE_DIR})
        endif()

        list(APPEND GRAPH_BENCHMARK_TARGETS_CUDA ${target})
    endforeach()

    # NOTE: the following parameters are just for performance debugging
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
    # target_compile_options( tnl-benchmark-graphs PUBLIC --resource-usage )
    # set_target_properties( tnl-benchmark-graphs PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )

    install(TARGETS ${GRAPH_BENCHMARK_TARGETS_CUDA} RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

if(TNL_BUILD_HIP)
    set(GRAPH_BENCHMARK_TARGETS_HIP)

    foreach(bench IN LISTS GRAPH_BENCHMARKS)
        set(target tnl-benchmark-graphs-${bench}-hip)
        add_executable(${target} tnl-benchmark-graphs-${bench}.hip)
        target_link_libraries(${target} PUBLIC TNL::TNL)

        if(Boost_GRAPH_FOUND)
            target_link_libraries(${target} PUBLIC Boost::graph)
            target_compile_definitions(${target} PUBLIC "-DHAVE_BOOST")
        endif()

        list(APPEND GRAPH_BENCHMARK_TARGETS_HIP ${target})
    endforeach()

    install(TARGETS ${GRAPH_BENCHMARK_TARGETS_HIP} RUNTIME DESTINATION bin COMPONENT benchmarks)
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
