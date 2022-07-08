if( BUILD_CUDA )
   cuda_add_executable( tnl-benchmark-linear-solvers-cuda tnl-benchmark-linear-solvers.cu
                        OPTIONS -DHAVE_CUSPARSE )
   target_link_libraries( tnl-benchmark-linear-solvers-cuda ${CUDA_cusparse_LIBRARY} )
   install( TARGETS tnl-benchmark-linear-solvers-cuda RUNTIME DESTINATION bin )

   if( MPI_CXX_FOUND AND ${WITH_MPI} )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-linear-solvers-cuda MPI::MPI_CXX )
   endif()
endif()

add_executable( tnl-benchmark-linear-solvers tnl-benchmark-linear-solvers.cpp )
install( TARGETS tnl-benchmark-linear-solvers RUNTIME DESTINATION bin )

if( MPI_CXX_FOUND AND ${WITH_MPI} )
   # enable MPI support in TNL
   target_compile_definitions( tnl-benchmark-linear-solvers PUBLIC "-DHAVE_MPI" )
   # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
   target_link_libraries( tnl-benchmark-linear-solvers MPI::MPI_CXX )
endif()
