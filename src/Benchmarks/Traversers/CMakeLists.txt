if( BUILD_CUDA )
   cuda_add_executable( tnl-benchmark-traversers-cuda tnl-benchmark-traversers.cu )

   install( TARGETS tnl-benchmark-traversers-cuda RUNTIME DESTINATION bin )
endif()

add_executable( tnl-benchmark-traversers tnl-benchmark-traversers.cpp )

install( TARGETS tnl-benchmark-traversers RUNTIME DESTINATION bin )
