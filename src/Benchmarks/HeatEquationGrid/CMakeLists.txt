if (BUILD_CUDA)
   CUDA_ADD_EXECUTABLE( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cu )
ELSE()
   add_executable( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cpp )
ENDIF()

install( TARGETS tnl-benchmark-heat-equation RUNTIME DESTINATION bin )
