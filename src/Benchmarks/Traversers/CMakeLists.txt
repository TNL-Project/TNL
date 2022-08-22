if( TNL_BUILD_CUDA )
   cuda_add_executable( tnl-benchmark-traversers-cuda tnl-benchmark-traversers.cu )
   install( TARGETS tnl-benchmark-traversers-cuda RUNTIME DESTINATION bin )
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-traversers tnl-benchmark-traversers.cpp )
   install( TARGETS tnl-benchmark-traversers RUNTIME DESTINATION bin )
endif()
