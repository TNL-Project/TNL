if(TNL_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# built only for one backend (GPU if available)
tnl_add_backend_executables(
    tnl-benchmark-dense-linear-solvers
    SOURCES tnl-benchmark-dense-linear-solvers.h
    BACKENDS CUDA HIP HOST
    SINGLE
    CUDA_LINK_LIBRARIES CUDA::cusolver
)

install(TARGETS tnl-benchmark-dense-linear-solvers RUNTIME DESTINATION bin COMPONENT benchmarks)
