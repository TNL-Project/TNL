if(TNL_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-benchmark-dense-linear-solvers
    SOURCES tnl-benchmark-dense-linear-solvers.h
    BACKENDS CUDA HIP HOST
    OPTIONAL_HOST
    CUDA_LINK_LIBRARIES CUDA::cusolver
    OUTPUT_TARGETS BENCHMARKS
)

install(TARGETS ${BENCHMARKS} RUNTIME DESTINATION bin COMPONENT benchmarks)
