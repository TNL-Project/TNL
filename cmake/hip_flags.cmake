# set project-specific (i.e. not exported) build options
set( CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall" )
# NOTE: cmake mirrors the hipcc.pl script and passes some flags to rocm-llvm
#       (--offload-arch=gfx803 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false)
#       which do not work with -O0 (leads to Memory access fault), hence -O1
set( CMAKE_HIP_FLAGS_DEBUG "-O1 -g -DHIP_ENABLE_PRINTF" )
set( CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG" )
set( CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELEASE} ${CMAKE_HIP_FLAGS_DEBUG}" )

if( TNL_USE_CI_FLAGS )
   # enforce (more or less) warning-free builds
   set( CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations" )
endif()

# optimize Release builds for the native CPU arch, unless explicitly disabled
if( TNL_USE_MARCH_NATIVE_FLAG )
   set( CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} -march=native -mtune=native" )
endif()

# force colorized output (the automatic detection in compilers does not work with Ninja)
target_compile_options( TNL_HIP INTERFACE
      $<$<HIP_COMPILER_ID:Clang>:-fcolor-diagnostics> ;
)
