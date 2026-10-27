# Value types of the scan benchmark. To speed up the build and reduce the peak
# memory of the compiler, the benchmark for each value type is instantiated in
# a separate source file generated from scan-benchmark-instance.in, so the value
# types are compiled in parallel and then linked into one executable.
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

function(tnl_add_scan_benchmark target extension)
    set(sources tnl-benchmark-scan.${extension})
    foreach(name IN LISTS SCAN_VALUE_TYPES)
        set(SCAN_VALUE_TYPE "${SCAN_CXX_TYPE_${name}}")
        set(source "${CMAKE_CURRENT_BINARY_DIR}/${target}-${name}.${extension}")
        configure_file(scan-benchmark-instance.in ${source} @ONLY)
        list(APPEND sources ${source})
    endforeach()
    add_executable(${target} ${sources})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PUBLIC TNL::TNL)
    install(TARGETS ${target} RUNTIME DESTINATION bin COMPONENT benchmarks)
endfunction()

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

set(BENCHMARKS)

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    tnl_add_scan_benchmark(tnl-benchmark-scan cpp)
    list(APPEND BENCHMARKS tnl-benchmark-scan)
    if(Thrust_FOUND)
        target_compile_definitions(tnl-benchmark-scan PUBLIC -DHAVE_THRUST)
        target_link_libraries(tnl-benchmark-scan PUBLIC ThrustHost)
    endif()
endif()

if(TNL_BUILD_CUDA)
    tnl_add_scan_benchmark(tnl-benchmark-scan-cuda cu)
    list(APPEND BENCHMARKS tnl-benchmark-scan-cuda)
    find_package(CCCL REQUIRED)
    target_link_libraries(tnl-benchmark-scan-cuda PUBLIC CCCL::CCCL)
    if(Thrust_FOUND)
        target_compile_definitions(tnl-benchmark-scan-cuda PUBLIC -DHAVE_THRUST)
        target_link_libraries(tnl-benchmark-scan-cuda PUBLIC ThrustCuda)
    endif()
endif()

if(TNL_BUILD_HIP)
    # the HIP benchmark compares TNL with hipCUB and rocPRIM, so it is skipped
    # (instead of failing the configuration) when they are not installed
    find_package(rocprim)
    find_package(hipcub)
    if(rocprim_FOUND AND hipcub_FOUND)
        tnl_add_scan_benchmark(tnl-benchmark-scan-hip hip)
        list(APPEND BENCHMARKS tnl-benchmark-scan-hip)
        target_link_libraries(tnl-benchmark-scan-hip PUBLIC roc::rocprim hip::hipcub)
        # rocThrust provides Thrust for HIP
        find_package(rocthrust)
        if(rocthrust_FOUND)
            target_compile_definitions(tnl-benchmark-scan-hip PUBLIC -DHAVE_THRUST)
            target_link_libraries(tnl-benchmark-scan-hip PUBLIC roc::rocthrust)
        endif()
    else()
        message(STATUS "rocPRIM or hipCUB not found, tnl-benchmark-scan-hip will not be built")
    endif()
endif()

if(TBB_FOUND)
    foreach(target IN ITEMS ${BENCHMARKS})
        target_compile_definitions(${target} PUBLIC -DHAVE_TBB)
        target_link_libraries(${target} PUBLIC TBB::tbb)
    endforeach()
endif()
