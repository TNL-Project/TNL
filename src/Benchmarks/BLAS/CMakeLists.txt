# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-blas tnl-benchmark-blas.cpp )
   target_link_libraries( tnl-benchmark-blas PUBLIC TNL::TNL_CXX )
   set( BENCHMARKS tnl-benchmark-blas )
endif()

if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-blas-cuda tnl-benchmark-blas.cu )
   target_link_libraries( tnl-benchmark-blas-cuda PUBLIC TNL::TNL_CUDA )
   find_package( CUDAToolkit REQUIRED )
   target_link_libraries( tnl-benchmark-blas-cuda PUBLIC CUDA::cublas )
   set( BENCHMARKS ${BENCHMARKS} tnl-benchmark-blas-cuda )
endif()

find_library( CBLAS_LIBRARY NAMES cblas )

# fallback for Centos 7.5 - libcblas.so does not exist, link to libtatlas.so or libsatlas.so
# https://forums.centos.org/viewtopic.php?t=48543
find_library( TATLAS_LIBRARY NAMES tatlas
              PATH_SUFFIXES atlas )
find_library( SATLAS_LIBRARY NAMES satlas
              PATH_SUFFIXES atlas )

find_package( BLAS )

if( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   find_path(BLAS_INCLUDE_DIRS cblas.h
   /usr/include
   /usr/local/include
   $ENV{BLAS_HOME}/include
   /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers )
endif()

foreach( target IN ITEMS ${BENCHMARKS} )
   if( CBLAS_LIBRARY )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
      target_link_libraries( ${target} PUBLIC ${CBLAS_LIBRARY} )
      if( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
         target_include_directories( ${target} PUBLIC "${BLAS_INCLUDE_DIRS}" )
      endif()
   elseif( TATLAS_LIBRARY )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
      target_link_libraries( ${target} PUBLIC ${TATLAS_LIBRARY} )
   elseif( SATLAS_LIBRARY )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
      target_link_libraries( ${target} PUBLIC ${SATLAS_LIBRARY} )
   else()
      # FIXME: We require the CBLAS interface, but CMake's FindBLAS cannot detect that,
      #        so this fails unless the BLAS implementation includes it in the same
      #        shared library file as the Fortran implementation (e.g. OpenBLAS does that).
      if( BLAS_FOUND )
         target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
         target_link_libraries( ${target} PUBLIC ${BLAS_LIBRARIES} )
      endif()
   endif()

   install( TARGETS ${target} RUNTIME DESTINATION bin )
endforeach()
