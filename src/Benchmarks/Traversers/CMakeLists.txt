if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-traversers-cuda tnl-benchmark-traversers.cu )
   target_link_libraries( tnl-benchmark-traversers-cuda PUBLIC TNL::TNL_CUDA )
   install( TARGETS tnl-benchmark-traversers-cuda RUNTIME DESTINATION bin )
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-traversers tnl-benchmark-traversers.cpp )
   target_link_libraries( tnl-benchmark-traversers PUBLIC TNL::TNL_CXX )
   install( TARGETS tnl-benchmark-traversers RUNTIME DESTINATION bin )
endif()
