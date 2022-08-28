if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cu )
   target_link_libraries( tnl-benchmark-heat-equation PUBLIC TNL::TNL_CUDA )
else()
   add_executable( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cpp )
   target_link_libraries( tnl-benchmark-heat-equation PUBLIC TNL::TNL_CXX )
endif()

install( TARGETS tnl-benchmark-heat-equation RUNTIME DESTINATION bin )
