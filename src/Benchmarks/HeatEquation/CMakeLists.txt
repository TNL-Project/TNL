# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-benchmark-heat-equation
    SOURCES tnl-benchmark-heat-equation.h
    BACKENDS CUDA HIP HOST
    OPTIONAL_HOST
    OUTPUT_TARGETS BENCHMARKS
)
# NOTE: the following parameters are just for performance debugging of the CUDA build
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
#target_compile_options( tnl-benchmark-heat-equation-cuda PUBLIC --resource-usage )
#set_target_properties( tnl-benchmark-heat-equation-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )

install(TARGETS ${BENCHMARKS} RUNTIME DESTINATION bin COMPONENT benchmarks)
