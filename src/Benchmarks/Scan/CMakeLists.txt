# find TBB (for C++17 parallel algorithms) https://stackoverflow.com/a/55989883
find_package(TBB)

# find Thrust (provided by recent CUDA toolkits)
set(CCCL_ENABLE_DEFAULT_THRUST_TARGET OFF)
find_package(CCCL COMPONENTS Thrust)
if(Thrust_FOUND)
    # https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md
    if(TBB_FOUND)
        set(_thrust_host_device TBB)
    elseif(TNL_USE_OPENMP)
        set(_thrust_host_device OMP)
    else()
        set(_thrust_host_device CPP)
    endif()
    if(NOT TARGET ThrustHost)
        thrust_create_target(ThrustHost HOST ${_thrust_host_device} DEVICE ${_thrust_host_device})
    endif()
    if(NOT TARGET ThrustCuda)
        thrust_create_target(ThrustCuda HOST CPP DEVICE CUDA) # NOTE: OMP does not work with CUDA
    endif()
endif()

set(SCAN_BACKENDS HOST)

if(TNL_BUILD_CUDA)
    find_package(CCCL REQUIRED)
    list(APPEND SCAN_BACKENDS CUDA)
endif()

if(TNL_BUILD_HIP)
    # the HIP benchmark compares TNL with hipCUB and rocPRIM, so it is skipped
    # (instead of failing the configuration) when they are not installed
    find_package(rocprim)
    find_package(hipcub)
    if(rocprim_FOUND AND hipcub_FOUND)
        list(APPEND SCAN_BACKENDS HIP)
        # rocThrust provides Thrust for HIP
        find_package(rocthrust)
    else()
        message(STATUS "rocPRIM or hipCUB not found, tnl-benchmark-scan-hip will not be built")
    endif()
endif()

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-benchmark-scan
    SOURCES tnl-benchmark-scan.h
    BACKENDS ${SCAN_BACKENDS}
    OPTIONAL_HOST
    CUDA_LINK_LIBRARIES CCCL::CCCL
    HIP_LINK_LIBRARIES roc::rocprim hip::hipcub
    OUTPUT_TARGETS BENCHMARKS
    OUTPUT_BACKENDS BENCHMARK_BACKENDS
)

# The benchmark for each value type is instantiated in a separate source file
# generated from scan-benchmark-instance.in, so the value types are compiled in
# parallel. With nvcc, this makes the build of the benchmark alone about 3.8
# times faster (87 s instead of 330 s on 8 cores) and reduces the peak memory
# of the compiler from 4.4 GB to 1.4 GB per process, at the cost of about 28 %
# more CPU time in total.
set(SCAN_VALUE_TYPES
    char
    short
    int
    long
    float
    double
    double2
)
set(SCAN_CXX_TYPE_char "char")
set(SCAN_CXX_TYPE_short "short")
set(SCAN_CXX_TYPE_int "int")
set(SCAN_CXX_TYPE_long "long")
set(SCAN_CXX_TYPE_float "float")
set(SCAN_CXX_TYPE_double "double")
set(SCAN_CXX_TYPE_double2 "TNL::Containers::StaticVector< 2, double >")
set(SCAN_EXTENSION_HOST cpp)
set(SCAN_EXTENSION_CUDA cu)
set(SCAN_EXTENSION_HIP hip)

foreach(target backend IN ZIP_LISTS BENCHMARKS BENCHMARK_BACKENDS)
    foreach(name IN LISTS SCAN_VALUE_TYPES)
        set(SCAN_VALUE_TYPE "${SCAN_CXX_TYPE_${name}}")
        set(source "${CMAKE_CURRENT_BINARY_DIR}/${target}-${name}.${SCAN_EXTENSION_${backend}}")
        configure_file(scan-benchmark-instance.in ${source} @ONLY)
        target_sources(${target} PRIVATE ${source})
    endforeach()
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    install(TARGETS ${target} RUNTIME DESTINATION bin COMPONENT benchmarks)

    if(backend STREQUAL "HOST" AND Thrust_FOUND)
        target_compile_definitions(${target} PUBLIC -DHAVE_THRUST)
        target_link_libraries(${target} PUBLIC ThrustHost)
    elseif(backend STREQUAL "CUDA" AND Thrust_FOUND)
        target_compile_definitions(${target} PUBLIC -DHAVE_THRUST)
        target_link_libraries(${target} PUBLIC ThrustCuda)
    elseif(backend STREQUAL "HIP" AND rocthrust_FOUND)
        target_compile_definitions(${target} PUBLIC -DHAVE_THRUST)
        target_link_libraries(${target} PUBLIC roc::rocthrust)
    endif()

    if(TBB_FOUND)
        target_compile_definitions(${target} PUBLIC -DHAVE_TBB)
        target_link_libraries(${target} PUBLIC TBB::tbb)
    endif()
endforeach()
