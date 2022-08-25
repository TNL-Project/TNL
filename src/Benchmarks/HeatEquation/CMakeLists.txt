if( TNL_BUILD_CUDA )
#   add_executable( tnl-benchmark-heat-equation-cuda tnl-benchmark-heat-equation.cu )
   add_executable( tnl-benchmark-simple-heat-equation-cuda tnl-benchmark-simple-heat-equation.cu )
   target_link_libraries( tnl-benchmark-simple-heat-equation-cuda PUBLIC TNL::TNL_CUDA )

   install( TARGETS
#               tnl-benchmark-heat-equation-cuda
               tnl-benchmark-simple-heat-equation-cuda
            RUNTIME DESTINATION bin )
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
#   add_executable( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cpp )
   add_executable( tnl-benchmark-simple-heat-equation tnl-benchmark-simple-heat-equation.cpp )
   target_link_libraries( tnl-benchmark-simple-heat-equation PUBLIC TNL::TNL_CXX )

   install( TARGETS
#               tnl-benchmark-heat-equation
               tnl-benchmark-simple-heat-equation
            RUNTIME DESTINATION bin )
endif()
