if( BUILD_CUDA )
#   cuda_add_executable( tnl-benchmark-heat-equation-cuda tnl-benchmark-heat-equation.cu )
   cuda_add_executable( tnl-benchmark-simple-heat-equation-cuda tnl-benchmark-simple-heat-equation.cu )

   install( TARGETS
#               tnl-benchmark-heat-equation-cuda
               tnl-benchmark-simple-heat-equation-cuda
            RUNTIME DESTINATION bin )
endif()

#add_executable( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cpp )
add_executable( tnl-benchmark-simple-heat-equation tnl-benchmark-simple-heat-equation.cpp )

install( TARGETS
#            tnl-benchmark-heat-equation
            tnl-benchmark-simple-heat-equation
         RUNTIME DESTINATION bin )
