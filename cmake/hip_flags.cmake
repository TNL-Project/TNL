# set project-specific (i.e. not exported) build options
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall")
# NOTE: cmake mirrors the hipcc.pl script and passes some flags to rocm-llvm
#       (--offload-arch=gfx803 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false)
#       which do not work with -O0 (leads to Memory access fault), hence -O1
set(CMAKE_HIP_FLAGS_DEBUG "-O1 -g -DHIP_ENABLE_PRINTF")
set(CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG")
# NOTE: the Debug configuration cannot be concatenated into RelWithDebInfo:
#       its "-O1" would be appended after the Release "-O3" and the last option would win.
#       HIP_ENABLE_PRINTF is only needed when assertions are enabled (i.e. without NDEBUG),
#       so only the debug information is added here.
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELEASE} -g")

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

# ROCm 10.1 (rocm-llvm 24.0) generates invalid typed DWARF expressions
# (DIOpDeref with a non-pointer operand) for the heterogeneous DWARF extensions.
# The module verifier then aborts the device LTO link with
# "Broken module found: DIOpDeref requires input to be pointer typed",
# so the extensions have to be disabled when debug info is generated.
# See the workaround used by ROCm:
# https://github.com/ROCm/rocm-systems/commit/00238f6c365df815d616fa9c6928fe5d58a48cc6
include(CheckCompilerFlag)
check_compiler_flag(HIP "-gno-heterogeneous-dwarf" TNL_HIP_HAS_GNO_HETEROGENEOUS_DWARF)
if(TNL_HIP_HAS_GNO_HETEROGENEOUS_DWARF)
    set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -Xarch_device -gno-heterogeneous-dwarf")
endif()

# optimize Release builds for the native CPU arch, unless explicitly disabled
if(TNL_USE_MARCH_NATIVE_FLAG)
    set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} -march=native -mtune=native")
endif()

# force colorized output (the automatic detection in compilers does not work with Ninja)
target_compile_options(TNL INTERFACE $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-fcolor-diagnostics> ;)
