# built only for one backend (GPU if available)
tnl_add_backend_executables(tnl-benchmark-heat-equation SOURCES tnl-benchmark-heat-equation.h BACKENDS CUDA HIP HOST SINGLE)
# NOTE: the following parameters are just for performance debugging of the CUDA build
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
#target_compile_options( tnl-benchmark-heat-equation PUBLIC --resource-usage )
#set_target_properties( tnl-benchmark-heat-equation PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )

install(TARGETS tnl-benchmark-heat-equation RUNTIME DESTINATION bin COMPONENT benchmarks)
