# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-blas tnl-benchmark-blas.cpp )
   set( BENCHMARKS tnl-benchmark-blas )
endif()

if( TNL_BUILD_CUDA )
   #find_library( CUDADEVRT NAMES cudadevrt )
   cuda_add_executable( tnl-benchmark-blas-cuda tnl-benchmark-blas.cu )
   cuda_add_cublas_to_target( tnl-benchmark-blas-cuda )
   #target_link_libraries( tnl-benchmark-blas-cuda ${CUDADEVRT} )#${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudadevrt.a )

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

foreach( target IN ITEMS ${BENCHMARKS} )
   if( CBLAS_LIBRARY )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
      target_link_libraries( ${target} ${CBLAS_LIBRARY} )
   elseif( TATLAS_LIBRARY )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
      target_link_libraries( ${target} ${TATLAS_LIBRARY} )
   elseif( SATLAS_LIBRARY )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
      target_link_libraries( ${target} ${SATLAS_LIBRARY} )
   else()
      # FIXME: We require the CBLAS interface, but CMake's FindBLAS cannot detect that,
      #        so this fails unless the BLAS implementation includes it in the same
      #        shared library file as the Fortran implementation (e.g. OpenBLAS does that).
      if( BLAS_FOUND )
         target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
         target_link_libraries( ${target} ${BLAS_LIBRARIES} )
      endif()
   endif()

   install( TARGETS ${target} RUNTIME DESTINATION bin )
endforeach()
