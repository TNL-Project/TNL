if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-parallel-for-cuda tnl-benchmark-parallel-for.cu)
    target_link_libraries(tnl-benchmark-parallel-for-cuda PUBLIC TNL::TNL_CUDA)
    # NOTE: the following parameters are just for performance debugging
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
    #target_compile_options( tnl-benchmark-heat-equation PUBLIC --resource-usage )
    #set_target_properties( tnl-benchmark-heat-equation PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )
    install(TARGETS tnl-benchmark-parallel-for-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
else()
    add_executable(tnl-benchmark-parallel-for tnl-benchmark-parallel-for.cpp)
    target_link_libraries(tnl-benchmark-parallel-for PUBLIC TNL::TNL_CXX)
    install(TARGETS tnl-benchmark-parallel-for RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
