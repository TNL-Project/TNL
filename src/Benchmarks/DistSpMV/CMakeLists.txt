if( BUILD_CUDA )
   cuda_add_executable( tnl-benchmark-distributed-spmv-cuda tnl-benchmark-distributed-spmv.cu )
   install( TARGETS tnl-benchmark-distributed-spmv-cuda RUNTIME DESTINATION bin )

   if( MPI_CXX_FOUND AND ${WITH_MPI} )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-distributed-spmv-cuda PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-distributed-spmv-cuda MPI::MPI_CXX )
   endif()
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-distributed-spmv tnl-benchmark-distributed-spmv.cpp )
   install( TARGETS tnl-benchmark-distributed-spmv RUNTIME DESTINATION bin )

   if( MPI_CXX_FOUND AND ${WITH_MPI} )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-distributed-spmv PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-distributed-spmv MPI::MPI_CXX )
   endif()
endif()
