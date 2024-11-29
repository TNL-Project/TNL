# set project-specific (i.e. not exported) build options
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall")
# NOTE: cmake mirrors the hipcc.pl script and passes some flags to rocm-llvm
#       (--offload-arch=gfx803 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false)
#       which do not work with -O0 (leads to Memory access fault), hence -O1
set(CMAKE_HIP_FLAGS_DEBUG "-O1 -g -DHIP_ENABLE_PRINTF")
set(CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELEASE} ${CMAKE_HIP_FLAGS_DEBUG}")

if(TNL_USE_CI_FLAGS)
    # enforce (more or less) warning-free builds
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations")
    # rocm-llvm 6.2.2 prints warnings due to unused variables:
    #   In file included from /opt/rocm/include/hip/hip_runtime.h:62:
    #   In file included from /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:119:
    #   /opt/rocm/include/hip/amd_detail/texture_indirect_functions.h:44:5: warning: unused variable 's' [-Wunused-variable]
    #      44 |     TEXTURE_OBJECT_PARAMETERS_INIT
    #         |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   /opt/rocm/include/hip/amd_detail/texture_indirect_functions.h:37:42: note: expanded from macro 'TEXTURE_OBJECT_PARAMETERS_INIT'
    #      37 |     unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;
    #         |                                          ^
    #   /opt/rocm/include/hip/amd_detail/texture_indirect_functions.h:335:5: warning: unused variable 's' [-Wunused-variable]
    #     335 |     TEXTURE_OBJECT_PARAMETERS_INIT
    #         |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   /opt/rocm/include/hip/amd_detail/texture_indirect_functions.h:37:42: note: expanded from macro 'TEXTURE_OBJECT_PARAMETERS_INIT'
    #      37 |     unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;
    #         |                                          ^
    #   /opt/rocm/include/hip/amd_detail/texture_indirect_functions.h:463:5: warning: unused variable 's' [-Wunused-variable]
    #     463 |     TEXTURE_OBJECT_PARAMETERS_INIT
    #         |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   /opt/rocm/include/hip/amd_detail/texture_indirect_functions.h:37:42: note: expanded from macro 'TEXTURE_OBJECT_PARAMETERS_INIT'
    #      37 |     unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;
    #         |                                          ^
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem /opt/rocm/include")
endif()

# optimize Release builds for the native CPU arch, unless explicitly disabled
if(TNL_USE_MARCH_NATIVE_FLAG)
    set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} -march=native -mtune=native")
endif()

# force colorized output (the automatic detection in compilers does not work with Ninja)
target_compile_options(TNL INTERFACE $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-fcolor-diagnostics> ;)
