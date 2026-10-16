# Optional external libraries:
#   Ginkgo  - set Ginkgo_DIR or CMAKE_PREFIX_PATH
#   PETSc   - found by pkg-config (set PKG_CONFIG_PATH or CMAKE_PREFIX_PATH)
#   Hypre   - HYPRE_ROOT points to a CPU build of Hypre (used by the host executable),
#             HYPRE_GPU_ROOT points to a CUDA or HIP build of Hypre (used by the GPU executable)
find_package(Ginkgo QUIET)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PETSC QUIET IMPORTED_TARGET PETSc)
endif()

set(HYPRE_GPU_ROOT "" CACHE PATH "Path to a GPU (CUDA or HIP) build of Hypre for the matrix assembly benchmark")

# Finds a Hypre installation in the given root directory and checks whether it is a GPU build.
function(tnl_find_hypre_variant prefix root)
    if(NOT root)
        return()
    endif()
    find_path(${prefix}_INCLUDE_DIR NAMES HYPRE.h HINTS ${root}/include ${root}/include/hypre NO_DEFAULT_PATH)
    find_library(${prefix}_LIBRARY NAMES HYPRE HINTS ${root}/lib ${root}/lib64 NO_DEFAULT_PATH)
    if(${prefix}_INCLUDE_DIR AND ${prefix}_LIBRARY)
        file(STRINGS ${${prefix}_INCLUDE_DIR}/HYPRE_config.h _gpu REGEX "^#define HYPRE_USING_GPU")
        if(_gpu)
            set(${prefix}_USES_GPU TRUE PARENT_SCOPE)
        else()
            set(${prefix}_USES_GPU FALSE PARENT_SCOPE)
        endif()
        set(${prefix}_FOUND TRUE PARENT_SCOPE)
        message(STATUS "Matrix assembly benchmark: found Hypre in ${root}")
    endif()
endfunction()

tnl_find_hypre_variant(HYPRE_CPU "${HYPRE_ROOT}")
tnl_find_hypre_variant(HYPRE_GPU "${HYPRE_GPU_ROOT}")

# Adds the optional libraries to the benchmark target.
function(tnl_matrix_assembly_setup target is_gpu)
    target_link_libraries(${target} PUBLIC TNL::TNL)
    install(TARGETS ${target} RUNTIME DESTINATION bin COMPONENT benchmarks)

    # FIXME: we can't link to ginkgo when using LLVM's libc++: https://github.com/ginkgo-project/ginkgo/issues/1571
    if(Ginkgo_FOUND)
        if(
            (NOT is_gpu AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            OR (is_gpu AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            OR (is_gpu AND TNL_BUILD_HIP)
        )
            target_compile_definitions(${target} PUBLIC "-DHAVE_GINKGO")
            target_link_libraries(${target} PUBLIC Ginkgo::ginkgo)
        endif()
    endif()

    # Hypre and PETSc need MPI
    if(NOT TNL_BUILD_MPI)
        return()
    endif()
    target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
    target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

    if(is_gpu AND HYPRE_GPU_FOUND AND HYPRE_GPU_USES_GPU)
        set(_hypre_include ${HYPRE_GPU_INCLUDE_DIR})
        set(_hypre_library ${HYPRE_GPU_LIBRARY})
    elseif(NOT is_gpu AND HYPRE_CPU_FOUND AND NOT HYPRE_CPU_USES_GPU)
        set(_hypre_include ${HYPRE_CPU_INCLUDE_DIR})
        set(_hypre_library ${HYPRE_CPU_LIBRARY})
    endif()
    if(_hypre_library)
        target_compile_definitions(${target} PUBLIC "-DHAVE_HYPRE")
        target_include_directories(${target} SYSTEM PUBLIC ${_hypre_include})
        target_link_libraries(${target} PUBLIC ${_hypre_library} -lm)
        if(is_gpu AND TNL_BUILD_CUDA)
            target_link_libraries(
                ${target}
                PUBLIC CUDA::cusparse CUDA::curand CUDA::cublas CUDA::cublasLt CUDA::cusolver CUDA::cudart
            )
        endif()
    endif()

    if(PETSC_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_PETSC")
        target_link_libraries(${target} PUBLIC PkgConfig::PETSC)
    endif()
endfunction()

if(TNL_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(tnl-benchmark-matrix-assembly-cuda tnl-benchmark-matrix-assembly.cu)
    target_link_libraries(tnl-benchmark-matrix-assembly-cuda PUBLIC CUDA::cusparse)
    tnl_matrix_assembly_setup(tnl-benchmark-matrix-assembly-cuda TRUE)
endif()

if(TNL_BUILD_HIP)
    add_executable(tnl-benchmark-matrix-assembly-hip tnl-benchmark-matrix-assembly.hip)
    tnl_matrix_assembly_setup(tnl-benchmark-matrix-assembly-hip TRUE)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-matrix-assembly tnl-benchmark-matrix-assembly.cpp)
    tnl_matrix_assembly_setup(tnl-benchmark-matrix-assembly FALSE)
endif()
