add_subdirectory( Ginkgo )
add_subdirectory( Hypre )

set( CPP_TARGETS  tnl-optimize-ranks tnl-turbulence-generator )
set( CUDA_TARGETS  tnl-optimize-ranks-cuda tnl-turbulence-generator-cuda )

# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   foreach( target IN ITEMS ${CPP_TARGETS} )
      add_executable( ${target} ${target}.cpp )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
   endforeach()
   install( TARGETS ${CPP_TARGETS} RUNTIME DESTINATION bin COMPONENT examples )
endif()

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_TARGETS} )
      add_executable( ${target} ${target}.cu )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA )
   endforeach()
   install( TARGETS ${CUDA_TARGETS} RUNTIME DESTINATION bin COMPONENT examples )
endif()

if( TNL_BUILD_MPI )
   # enable MPI support in TNL and add MPI to the targets:
   # https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
   if( TNL_BUILD_CPP_TARGETS )
      target_compile_definitions( tnl-optimize-ranks PUBLIC "-DHAVE_MPI" )
      target_link_libraries( tnl-optimize-ranks PUBLIC MPI::MPI_CXX )
   endif()
   if( TNL_BUILD_CUDA )
      target_compile_definitions( tnl-optimize-ranks-cuda PUBLIC "-DHAVE_MPI" )
      target_link_libraries( tnl-optimize-ranks-cuda PUBLIC MPI::MPI_CXX )
   endif()
endif()
