set( CUDA_SAMPLES_DIR $ENV{CUDA_SAMPLES_DIR} )
if( NOT CUDA_SAMPLES_DIR STREQUAL "none" )
   set( CUDA_SAMPLES_FLAGS "-I${CUDA_SAMPLES_DIR} -DHAVE_CUDA_SAMPLES")
   message( "   CUDA_SAMPLES_FLAGS = ${CUDA_SAMPLES_FLAGS}" )
endif()

if( TNL_BUILD_CUDA )
   # Source code of reference algorithms contains warning which turn into errers with CI/CD compiler flags. Therefore we use -Wno-error to turn it off.
   cuda_add_executable( tnl-benchmark-sort-cuda tnl-benchmark-sort.cu OPTIONS -Xcompiler -Wno-error=switch,-Wno-error=sign-compare ${CUDA_SAMPLES_FLAGS} )
   target_link_libraries( tnl-benchmark-sort-cuda ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} )
   install( TARGETS tnl-benchmark-sort-cuda RUNTIME DESTINATION bin )
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-sort tnl-benchmark-sort.cpp )
   install( TARGETS tnl-benchmark-sort RUNTIME DESTINATION bin )
endif()
