find_package( Boost COMPONENTS graph )
if( Boost_GRAPH_FOUND )
   message( "Boost found ${Boost_INCLUDE_DIRS}")
   set( BOOST_FLAGS "-DHAVE_BOOST ${Boost_INCLUDE_DIRS}" )
else()
   if( Boost_FOUND )
      message( "Boost found ${Boost_INCLUDE_DIRS} but not Boost::Graph. Benchmarks with Boost::Graph will not be built.")
   else()
      message( "Boost not found. Benchmarks with Boost::Graph will not be built.")
   endif()
endif()

if( TNL_BUILD_CUDA )
   # Find Gunrock
   find_path(GUNROCK_INCLUDE_DIR gunrock/algorithms/bfs.hxx)
   if( GUNROCK_INCLUDE_DIR )
      message(STATUS "Gunrock found ${GUNROCK_INCLUDE_DIR}")
      set( GUNROCK_FLAGS -DHAVE_GUNROCK_ ${Gunrock_INCLUDE_DIR} -D SM_TARGET=86 ) # TODO: Gunrock is not working currently
   else()
         message(STATUS "Gunrock not found.")
   endif()
endif()

if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-graphs tnl-benchmark-graphs.cu )
   target_compile_options( tnl-benchmark-graphs PUBLIC ${GUNROCK_FLAGS} )
   target_link_libraries( tnl-benchmark-graphs PUBLIC TNL::TNL_CUDA )
   # NOTE: the following parameters are just for performance debugging
   # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
   #target_compile_options( tnl-benchmark-graphs PUBLIC --resource-usage )
   #set_target_properties( tnl-benchmark-graphs PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )
else()
   add_executable( tnl-benchmark-graphs tnl-benchmark-graphs.cpp )
   target_link_libraries( tnl-benchmark-graphs PUBLIC TNL::TNL_CXX )
endif()

install( TARGETS tnl-benchmark-graphs RUNTIME DESTINATION bin )
