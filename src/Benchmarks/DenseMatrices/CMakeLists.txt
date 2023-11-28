# Include the CUTLASS headers
include_directories(/home/krastale/.local/include/cutlass/include)

# Include the MAGMA headers
include_directories(/home/krastale/.local/include/magma/include)

# CUDA Setup
find_package(CUDAToolkit REQUIRED)

# TNL Build CPP Targets
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-densematrices tnl-benchmark-densematrices.cpp)
    target_link_libraries(tnl-benchmark-densematrices PUBLIC TNL::TNL)
    set(BENCHMARKS ${BENCHMARKS} tnl-benchmark-densematrices)
endif()

# TNL Build CUDA Targets
if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-densematrices-cuda tnl-benchmark-densematrices.cu)
    target_link_libraries(tnl-benchmark-densematrices-cuda PUBLIC TNL::TNL)
    target_include_directories(tnl-benchmark-densematrices-cuda PUBLIC /home/krastale/.local/include/cutlass/include)
    target_include_directories(tnl-benchmark-densematrices-cuda PUBLIC /home/krastale/.local/include/magma/include)
    target_link_libraries(tnl-benchmark-densematrices-cuda PUBLIC CUDA::cublas)

    # Linking MAGMA
    find_library(MAGMA_LIBRARY NAMES cutlass PATHS /home/krastale/.local/include/cutlass/build/lib)
    if(MAGMA_LIBRARY)
        target_link_libraries(tnl-benchmark-densematrices-cuda PUBLIC ${CUTLASS_LIBRARY})
        add_definitions(-DHAVE_CUTLASS)
    else()
        message(WARNING "CUTLASS library not found.")
    endif()

    # Linking Cutlass
    find_library(CUTLASS_LIBRARY NAMES magma PATHS /home/krastale/.local/include/magma/lib)
    if(MAGMA_LIBRARY)
        target_link_libraries(tnl-benchmark-densematrices-cuda PUBLIC ${MAGMA_LIBRARY})
        add_definitions(-DHAVE_MAGMA)
    else()
        message(WARNING "MAGMA library not found.")
    endif()

    set(BENCHMARKS ${BENCHMARKS} tnl-benchmark-densematrices-cuda)
endif()

# Installation
foreach(target IN ITEMS ${BENCHMARKS})
    install(TARGETS ${target} RUNTIME DESTINATION bin)
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


