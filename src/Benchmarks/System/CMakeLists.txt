add_executable( tnl-benchmark-memory-access tnl-benchmark-memory-access.cpp )
target_link_libraries( tnl-benchmark-memory-access PUBLIC TNL::TNL_CXX )


install( TARGETS tnl-benchmark-memory-access RUNTIME DESTINATION bin )
