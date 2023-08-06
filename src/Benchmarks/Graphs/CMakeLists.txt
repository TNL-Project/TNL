find_package( Boost OPTIONAL_COMPONENTS graph )
if( Boost_GRAPH_FOUND )
   message( "Boost found ${Boost_INCLUDE_DIRS}")
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

      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gunrock-test.cpp
      "#include <gunrock/algorithms/bfs.hxx>
      #include <gunrock/algorithms/sssp.hxx>
      int main() { return 0; }
      ")

      try_compile(
         GUNROCK_TEST
         ${CMAKE_CURRENT_BINARY_DIR}
         ${CMAKE_CURRENT_BINARY_DIR}/gunrock-test.cpp
         CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${GUNROCK_INCLUDE_DIR}"
         OUTPUT_VARIABLE GUNROCK_TEST_OUTPUT
      )
      if( GUNROCK_COMPILED )
         message(STATUS "Gunrock compiled successfully.")
         set( GUNROCK_FLAGS -DHAVE_GUNROCK ${Gunrock_INCLUDE_DIR} -D SM_TARGET=86 )
      else()
         message(STATUS "Gunrock failed to compile, will be omitted in the benchmark.")
      endif()
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

if( Boost_GRAPH_FOUND )
   target_link_libraries( tnl-benchmark-graphs PUBLIC Boost::graph )
   target_compile_definitions( tnl-benchmark-graphs PUBLIC "-DHAVE_BOOST" )
endif()

install( TARGETS tnl-benchmark-graphs RUNTIME DESTINATION bin )
