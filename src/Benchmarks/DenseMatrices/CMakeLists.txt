# Option to enable Tensor Core support explicitly
option(USE_TENSOR_CORES "Enable to compile with Tensor Core support" OFF)

# TNL Build CPP Targets
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-densematrices tnl-benchmark-densematrices.cpp)
    target_link_libraries(tnl-benchmark-densematrices PUBLIC TNL::TNL)
    set(BENCHMARKS ${BENCHMARKS} tnl-benchmark-densematrices)
endif()

# TNL Build CUDA Targets
if(TNL_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(tnl-benchmark-densematrices-cuda tnl-benchmark-densematrices.cu)
    target_link_libraries(tnl-benchmark-densematrices-cuda PUBLIC TNL::TNL)
    target_include_directories(tnl-benchmark-densematrices-cuda PUBLIC /opt/cuda/targets/x86_64-linux/include)
    target_include_directories(tnl-benchmark-densematrices-cuda PUBLIC /opt/cuda/targets/x86_64-linux/include)
    target_link_libraries(tnl-benchmark-densematrices-cuda PUBLIC CUDA::cublas)

    if(USE_TENSOR_CORES)
    message(STATUS "Compiling with Tensor Core support.")
    target_compile_definitions(tnl-benchmark-densematrices-cuda PRIVATE USE_TENSOR_CORES=1)
    target_compile_options(tnl-benchmark-densematrices-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code arch=compute_75,code=sm_75>
    )
    else()
    message(STATUS "Compiling without Tensor Core support.")
    target_compile_options(tnl-benchmark-densematrices-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code arch=compute_50,code=sm_50>
    )
    endif()

    # Linking MAGMA
    find_library(MAGMA_LIBRARY NAMES magma PATHS /opt/cuda/targets/x86_64-linux/lib)
    if(MAGMA_LIBRARY)
         target_link_libraries(tnl-benchmark-densematrices-cuda PUBLIC ${MAGMA_LIBRARY})
         add_definitions(-DHAVE_MAGMA)
    else()
         message(WARNING "MAGMA library not found.")
    endif()

    # Linking Cutlass
    find_package(NvidiaCutlass)
    if(NvidiaCutlass_FOUND)
      #target_link_libraries(tnl-benchmark-densematrices-cuda PUBLIC nvidia::cutlass::cutlass)
      target_compile_definitions(tnl-benchmark-densematrices-cuda PUBLIC -DHAVE_CUTLASS)
    else()
         message(WARNING "CUTLASS library not found.")
    endif()

    set(BENCHMARKS ${BENCHMARKS} tnl-benchmark-densematrices-cuda)
endif()

if( TNL_BUILD_HIP )
   add_executable( tnl-benchmark-densematrices-hip tnl-benchmark-densematrices.hip )
   target_link_libraries( tnl-benchmark-densematrices-hip PUBLIC TNL::TNL )
   find_package( HIPBLAS REQUIRED )
   target_link_libraries( tnl-benchmark-densematrices-hip PUBLIC roc::hipblas )
   set( BENCHMARKS ${BENCHMARKS} tnl-benchmark-densematrices-hip )
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
find_path(BLAS_INCLUDE_DIRS cblas.h
    /usr/include
    /usr/local/include
    $ENV{BLAS_HOME}/include
    /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers
    PATH_SUFFIXES openblas
)

if(BLAS_FOUND)
    if(BLAS_INCLUDE_DIRS)
        foreach(target IN ITEMS ${BENCHMARKS})
            target_include_directories(${target} PUBLIC "${BLAS_INCLUDE_DIRS}")
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



