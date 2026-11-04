# set necessary CUDA compiler flags on the interface
target_compile_options(
    TNL
    INTERFACE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
        --expt-relaxed-constexpr
        ;
        --extended-lambda
        ;
        --default-stream
        per-thread
        ;
        >
)

# Disable false compiler warnings
#   reference for the --diag_suppress and --display_error_number flags: https://stackoverflow.com/a/54142937
#   incomplete list of tokens: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
target_compile_options(
    TNL
    INTERFACE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
        -Wno-deprecated-gpu-targets
        ;
        "SHELL:-Xcudafe --diag_suppress=code_is_unreachable"
        ;
        "SHELL:-Xcudafe --diag_suppress=loop_not_reachable"
        ;
        "SHELL:-Xcudafe --diag_suppress=implicit_return_from_non_void_function"
        ;
        "SHELL:-Xcudafe --diag_suppress=unsigned_compare_with_zero"
        ;
        --display-error-number
        ;
        >
)
# This diagnostic is just plain wrong in CUDA 9 and later, see https://github.com/kokkos/kokkos/issues/1470
target_compile_options(
    TNL
    INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: "SHELL:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored" >
)
# nvcc 10 causes many invalid VLA errors in the host code
target_compile_options(
    TNL
    INTERFACE
        $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<VERSION_LESS:$<CUDA_COMPILER_VERSION>,11>>:
        "SHELL:-Xcompiler -Wno-vla"
        >
)

# set project-specific (i.e. not exported) build options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE} ${CMAKE_CUDA_FLAGS_DEBUG}")

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")
    if(TNL_USE_CI_FLAGS AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        # enforce (more or less) warning-free builds for host code
        set(CMAKE_CUDA_FLAGS
            "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror -Xcompiler -Wno-error=deprecated -Xcompiler -Wno-error=deprecated-declarations"
        )
    endif()

    if(DEFINED ENV{GITLAB_CI} AND CMAKE_CUDA_HOST_COMPILER_ID STREQUAL "NVHPC")
        # need to disable some optimizations to lower memory usage in CI
        set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")
    endif()
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # CUDA 13 moved the CCCL headers (e.g. <cuda/std/...>) from include/ to include/cccl/,
    # which CUDA toolkit detection in Clang 23 (--cuda-path) does not know about.
    # We need to extend the include path, otherwise headers like <cuda/std/type_traits> are not found.
    if(EXISTS "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include/cccl")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include/cccl")
    endif()
    if(TNL_USE_CI_FLAGS)
        # enforce (more or less) warning-free builds
        # -Wno-error=pass-failed due to optimizer noise in CCCL headers (e.g. non-unrolled loops in CUB radix sort) - not actionable
        set(CMAKE_CUDA_FLAGS
            "${CMAKE_CUDA_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=unknown-cuda-version -Wno-error=pass-failed"
        )
    endif()
    # CMake passes --no-cuda-include-ptx=<arch> (for CMAKE_CUDA_ARCHITECTURES) also in compile-only steps
    # with -fgpu-rdc, where the option has no effect because PTX embedding is decided at link time.
    # Clang reports it as an unused argument (https://github.com/llvm/llvm-project/issues/58491)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-error=unused-command-line-argument")

    # libstdc++ enables assertion-based hardening in unoptimized builds (since GCC 12)
    # and clang's CUDA wrapper for __glibcxx_assert_fail compiles to an unresolved abort
    # reference in device code; disable the hardening for the device compilation only
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xarch_device -D_GLIBCXX_NO_ASSERTIONS")

    # -O2 optimizes the device code, which clang does not do without an -O flag, unlike
    # nvcc which optimizes it unless -G is given -- without it, the CUDA tests run their
    # kernels at -O0 and are an order of magnitude slower
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -Xarch_device -O2")

    # Clang < 20 emits invalid PTX (ptxas fatal error) when the device code contains
    # debug info, see https://github.com/llvm/llvm-project/issues/58491. The fix was
    # merged after the 19 branch was created, so -g0 is still needed for older Clang.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 20)
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xarch_device -g0")
    endif()
endif()

# optimize Release builds for the native CPU arch, unless explicitly disabled
if(TNL_USE_MARCH_NATIVE_FLAG)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -march=native -mtune=native")
        #elseif( CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" )
        # setting the correct flag for the nvcc's host compiler would be too complicated...
    endif()
endif()

# force colorized output (the automatic detection in compilers does not work with Ninja)
target_compile_options(
    TNL
    INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-fcolor-diagnostics> ;
    # nvcc does not support colored diagnostics
    #$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:> ;
)
