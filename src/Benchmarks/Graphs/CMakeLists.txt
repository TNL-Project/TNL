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

# Find the GAP Benchmark Suite reference code (gapbs, https://github.com/sbeamer/gapbs), whose BFS, SSSP and CC
# kernels serve as the CPU baselines. It is not a library, its kernels are compiled from its sources in a separate
# translation unit with the host compiler.
set(GAPBS_BENCHMARKS bfs sssp cc)
find_path(
    GAPBS_SOURCE_DIR
    NAMES bfs.cc sssp.cc cc.cc benchmark.h
    PATHS ${GAPBS_ROOT} $ENV{GAPBS_ROOT} $ENV{HOME}/install/gapbs
    PATH_SUFFIXES src
    NO_DEFAULT_PATH
)
find_package(OpenMP COMPONENTS CXX)
if(GAPBS_SOURCE_DIR AND OpenMP_CXX_FOUND)
    message(STATUS "gapbs found: ${GAPBS_SOURCE_DIR}")
    add_library(tnl-benchmark-graphs-gapbs STATIC GapbsKernels.cpp)
    target_include_directories(tnl-benchmark-graphs-gapbs SYSTEM PRIVATE ${GAPBS_SOURCE_DIR})
    target_compile_features(tnl-benchmark-graphs-gapbs PRIVATE cxx_std_17)
    # the gapbs sources are not ours, do not report their warnings
    target_compile_options(tnl-benchmark-graphs-gapbs PRIVATE -w)
    target_link_libraries(tnl-benchmark-graphs-gapbs PUBLIC OpenMP::OpenMP_CXX)
    set(GAPBS_FOUND TRUE)
else()
    message(STATUS "gapbs not found (set GAPBS_ROOT). Benchmarks with gapbs will be omitted.")
endif()

# Adds the gapbs kernels to a benchmark target if they are used by the benchmark.
function(tnl_graph_benchmark_add_gapbs target bench)
    if(GAPBS_FOUND AND bench IN_LIST GAPBS_BENCHMARKS)
        target_link_libraries(${target} PUBLIC tnl-benchmark-graphs-gapbs)
        target_compile_definitions(${target} PUBLIC "-DHAVE_GAPBS")
    endif()
endfunction()

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

    foreach(target IN LISTS targets)
        tnl_graph_benchmark_add_gapbs(${target} ${bench})
    endforeach()

    # options and libraries for the CUDA variants
    foreach(target backend IN ZIP_LISTS targets backends)
        if(NOT backend STREQUAL "CUDA")
            continue()
        endif()
        if(GUNROCK_FOUND AND NOT bench STREQUAL "mst")
            target_compile_definitions(${target} PUBLIC ${GUNROCK_DEFINITIONS})
            target_include_directories(${target} SYSTEM PUBLIC ${GUNROCK_INCLUDE_DIR})
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
