if(TNL_BUILD_CUDA)
    find_package(CCCL REQUIRED)
endif()

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-benchmark-sort
    SOURCES tnl-benchmark-sort.h
    OPTIONAL_HOST
    CUDA_LINK_LIBRARIES CCCL::CCCL
    OUTPUT_TARGETS BENCHMARKS
)

# Add include directories from CUDA samples for reference algorithms
if(TARGET tnl-benchmark-sort-cuda AND CUDA_SAMPLES_FOUND)
    target_include_directories(tnl-benchmark-sort-cuda SYSTEM PRIVATE ${CUDA_SAMPLES_INCLUDE_DIRS})
    target_compile_definitions(tnl-benchmark-sort-cuda PRIVATE HAVE_CUDA_SAMPLES)
endif()

if(BENCHMARKS)
    install(TARGETS ${BENCHMARKS} RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
