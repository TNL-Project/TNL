# set necessary CUDA compiler flags on the interface
target_compile_options( TNL_CUDA INTERFACE
      $<$<CUDA_COMPILER_ID:NVIDIA>:
            --expt-relaxed-constexpr ;
            --extended-lambda ;
            --default-stream per-thread ;
      >
)

# Disable false compiler warnings
#   reference for the --diag_suppress and --display_error_number flags: https://stackoverflow.com/a/54142937
#   incomplete list of tokens: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
target_compile_options( TNL_CUDA INTERFACE
      $<$<CUDA_COMPILER_ID:NVIDIA>:
            -Wno-deprecated-gpu-targets ;
            "SHELL:-Xcudafe --diag_suppress=code_is_unreachable" ;
            "SHELL:-Xcudafe --diag_suppress=loop_not_reachable" ;
            "SHELL:-Xcudafe --diag_suppress=implicit_return_from_non_void_function" ;
            "SHELL:-Xcudafe --diag_suppress=unsigned_compare_with_zero" ;
            --display-error-number ;
      >
)
# This diagnostic is just plain wrong in CUDA 9 and later, see https://github.com/kokkos/kokkos/issues/1470
target_compile_options( TNL_CUDA INTERFACE
      $<$<CUDA_COMPILER_ID:NVIDIA>:
            "SHELL:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored"
      >
)
# nvcc 10 causes many invalid VLA errors in the host code
target_compile_options( TNL_CUDA INTERFACE
      $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<VERSION_LESS:$<CUDA_COMPILER_VERSION>,11>>:
            "SHELL:-Xcompiler -Wno-vla"
      >
)

# set project-specific (i.e. not exported) build options
set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall" )
set( CMAKE_CUDA_FLAGS_DEBUG "-g" )
set( CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG" )
set( CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE} ${CMAKE_CUDA_FLAGS_DEBUG}" )

if( CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" )
   set( CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info" )
   if( TNL_USE_CI_FLAGS AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" )
      # enforce (more or less) warning-free builds for host code
      set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror -Xcompiler -Wno-error=deprecated -Xcompiler -Wno-error=deprecated-declarations" )
   endif()
endif()

if( CMAKE_CUDA_COMPILER_ID STREQUAL "Clang" )
   if( TNL_USE_CI_FLAGS )
      # enforce (more or less) warning-free builds
      set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=unknown-cuda-version" )
   endif()
   # workaround for Clang 15
   # https://github.com/llvm/llvm-project/issues/58491
   set( CMAKE_CUDA_FLAGS_DEBUG "-g -Xarch_device -g0" )
endif()

# force colorized output (the automatic detection in compilers does not work with Ninja)
target_compile_options( TNL_CUDA INTERFACE
      $<$<CUDA_COMPILER_ID:Clang>:-fcolor-diagnostics> ;
      # nvcc does not support colored diagnostics
      #$<$<CUDA_COMPILER_ID:NVIDIA>:> ;
)
