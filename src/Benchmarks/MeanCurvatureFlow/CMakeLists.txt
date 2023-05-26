find_package( DECOMP REQUIRED )
target_link_libraries( TNL_CXX INTERFACE DECOMP::DECOMP_CXX )
target_link_libraries( TNL_CUDA INTERFACE DECOMP::DECOMP_CUDA )
find_package( CUDAToolkit REQUIRED )
target_link_libraries( TNL_CUDA INTERFACE CUDA::cublas CUDA::cusolver )

if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-mean-curvature-flow tnl-benchmark-mean-curvature-flow.cu )
   target_link_libraries( tnl-benchmark-mean-curvature-flow PUBLIC TNL::TNL_CUDA )
   target_link_libraries( tnl-benchmark-mean-curvature-flow PUBLIC DECOMP::DECOMP_CUDA )
   # NOTE: the following parameters are just for performance debugging
   # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
   #target_compile_options( tnl-benchmark-heat-equation PUBLIC --resource-usage )
   #set_target_properties( tnl-benchmark-heat-equation PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )
else()
   add_executable( tnl-benchmark-mean-curvature-flow tnl-benchmark-mean-curvature-flow.cpp )
   target_link_libraries( tnl-benchmark-mean-curvature-flow PUBLIC TNL::TNL_CXX )
   target_link_libraries( tnl-benchmark-mean-curvature-flow PUBLIC DECOMP::DECOMP_CXX )
endif()

install( TARGETS tnl-benchmark-mean-curvature-flow RUNTIME DESTINATION bin )
