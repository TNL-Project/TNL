# Option to enable Tensor Core support explicitly
option(USE_TENSOR_CORES "Enable to compile with Tensor Core support" OFF)

if(TNL_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(NvidiaCutlass)
    find_library(MAGMA_LIBRARY NAMES magma PATHS /opt/cuda/targets/x86_64-linux/lib)

    if(USE_TENSOR_CORES)
        message(STATUS "Compiling with Tensor Core support.")
    else()
        message(STATUS "Compiling without Tensor Core support.")
    endif()
    if(NOT MAGMA_LIBRARY)
        message(WARNING "MAGMA library not found.")
    endif()
    if(NOT NvidiaCutlass_FOUND)
        message(WARNING "CUTLASS library not found.")
    endif()
endif()

if(TNL_BUILD_HIP)
    find_package(HIPBLAS REQUIRED)
endif()

set(DENSE_MATRIX_BENCHMARKS
    tnl-benchmark-dense-matrix-multiplication
    tnl-benchmark-dense-matrix-transposition
    tnl-benchmark-dense-matrix-vector-product
)

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
foreach(bench IN LISTS DENSE_MATRIX_BENCHMARKS)
    tnl_add_backend_executables(
        ${bench}
        SOURCES ${bench}.h
        BACKENDS HOST CUDA HIP
        OPTIONAL_HOST
        CUDA_LINK_LIBRARIES CUDA::cublas
        HIP_LINK_LIBRARIES roc::hipblas
        OUTPUT_TARGETS targets
        OUTPUT_BACKENDS backends
    )

    # libraries and options for the CUDA variants
    foreach(target backend IN ZIP_LISTS targets backends)
        if(NOT backend STREQUAL "CUDA")
            continue()
        endif()
        target_include_directories(${target} PUBLIC /opt/cuda/targets/x86_64-linux/include)
        if(MAGMA_LIBRARY)
            target_link_libraries(${target} PUBLIC ${MAGMA_LIBRARY})
        endif()
        if(NvidiaCutlass_FOUND)
            target_compile_definitions(${target} PUBLIC -DHAVE_CUTLASS)
        endif()
        if(USE_TENSOR_CORES AND NOT bench MATCHES "vector-product")
            target_compile_definitions(${target} PRIVATE USE_TENSOR_CORES=1)
        endif()
    endforeach()

    list(APPEND BENCHMARKS ${targets})
endforeach()

if(TNL_BUILD_CUDA AND MAGMA_LIBRARY)
    add_definitions(-DHAVE_MAGMA)
endif()

# Installation
foreach(target IN ITEMS ${BENCHMARKS})
    install(TARGETS ${target} RUNTIME DESTINATION bin COMPONENT benchmarks)
endforeach()

# BLAS Libraries
find_library(CBLAS_LIBRARY NAMES cblas)
find_library(TATLAS_LIBRARY NAMES tatlas PATH_SUFFIXES atlas)
find_library(SATLAS_LIBRARY NAMES satlas PATH_SUFFIXES atlas)
find_package(BLAS)

# BLAS Include Directories
find_path(
    BLAS_INCLUDE_DIRS
    cblas.h
    /usr/include
    /usr/local/include
    $ENV{BLAS_HOME}/include
    /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers
    PATH_SUFFIXES openblas
)

if(BLAS_FOUND)
    if(BLAS_INCLUDE_DIRS)
        foreach(target IN ITEMS ${BENCHMARKS})
            target_include_directories(${target} SYSTEM PUBLIC "${BLAS_INCLUDE_DIRS}")
            if(CBLAS_LIBRARY)
                target_compile_definitions(${target} PUBLIC "-DHAVE_BLAS")
                target_link_libraries(${target} PUBLIC ${CBLAS_LIBRARY})
            elseif(TATLAS_LIBRARY)
                target_compile_definitions(${target} PUBLIC "-DHAVE_BLAS")
                target_link_libraries(${target} PUBLIC ${TATLAS_LIBRARY})
            elseif(SATLAS_LIBRARY)
                target_compile_definitions(${target} PUBLIC "-DHAVE_BLAS")
                target_link_libraries(${target} PUBLIC ${SATLAS_LIBRARY})
            else()
                target_compile_definitions(${target} PUBLIC "-DHAVE_BLAS")
                target_link_libraries(${target} PUBLIC BLAS::BLAS)
            endif()
        endforeach()
    else()
        message(WARNING "Could not find BLAS headers (cblas.h) - benchmarks with BLAS will not be built.")
    endif()
else()
    message(WARNING "Could not find BLAS - benchmarks with BLAS will not be built.")
endif()
