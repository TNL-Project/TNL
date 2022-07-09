if( BUILD_CUDA )
   cuda_add_executable( tnl-benchmark-ode-solvers-cuda tnl-benchmark-ode-solvers.cu
                        OPTIONS -DHAVE_CUSPARSE )
   target_link_libraries( tnl-benchmark-ode-solvers-cuda ${CUDA_cusparse_LIBRARY} )
   install( TARGETS tnl-benchmark-ode-solvers-cuda RUNTIME DESTINATION bin )

   if( MPI_CXX_FOUND AND ${WITH_MPI} )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-ode-solvers-cuda PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-ode-solvers-cuda MPI::MPI_CXX )
   endif()
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-ode-solvers tnl-benchmark-ode-solvers.cpp )
   install( TARGETS tnl-benchmark-ode-solvers RUNTIME DESTINATION bin )

   if( MPI_CXX_FOUND AND ${WITH_MPI} )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-ode-solvers PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-ode-solvers MPI::MPI_CXX )
   endif()
endif()
