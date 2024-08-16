# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-blas tnl-benchmark-blas.cpp)
    target_link_libraries(tnl-benchmark-blas PUBLIC TNL::TNL)
    set(BENCHMARKS tnl-benchmark-blas)
endif()

if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-blas-cuda tnl-benchmark-blas.cu)
    target_link_libraries(tnl-benchmark-blas-cuda PUBLIC TNL::TNL)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tnl-benchmark-blas-cuda PUBLIC CUDA::cublas)
    set(BENCHMARKS ${BENCHMARKS} tnl-benchmark-blas-cuda)
endif()

if(TNL_BUILD_HIP)
    add_executable(tnl-benchmark-blas-hip tnl-benchmark-blas.hip)
    target_link_libraries(tnl-benchmark-blas-hip PUBLIC TNL::TNL)
    find_package(HIPBLAS REQUIRED)
    target_link_libraries(tnl-benchmark-blas-hip PUBLIC roc::hipblas)
    set(BENCHMARKS ${BENCHMARKS} tnl-benchmark-blas-hip)
endif()

foreach(target IN ITEMS ${BENCHMARKS})
    install(TARGETS ${target} RUNTIME DESTINATION bin COMPONENT benchmarks)
endforeach()

find_library(CBLAS_LIBRARY NAMES cblas)

# fallback for Centos 7.5 - libcblas.so does not exist, link to libtatlas.so or libsatlas.so
# https://forums.centos.org/viewtopic.php?t=48543
find_library(TATLAS_LIBRARY NAMES tatlas PATH_SUFFIXES atlas)
find_library(SATLAS_LIBRARY NAMES satlas PATH_SUFFIXES atlas)

find_package(BLAS)

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
                # FIXME: We require the CBLAS interface, but CMake's FindBLAS cannot detect that,
                #        so this fails unless the BLAS implementation includes it in the same
                #        shared library file as the Fortran implementation (e.g. OpenBLAS does that).
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

# find TBB (for C++17 parallel algorithms) https://stackoverflow.com/a/55989883
find_package(TBB)
if(TBB_FOUND)
    foreach(target IN ITEMS ${BENCHMARKS})
        target_compile_definitions(${target} PUBLIC -DHAVE_TBB)
        target_link_libraries(${target} PUBLIC TBB::tbb)
    endforeach()
endif()

# find Thrust (provided by recent CUDA toolkits)
find_package(Thrust)
if(Thrust_FOUND)
    # https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md
    if(TBB_FOUND)
        set(_thrust_host_device TBB)
    elseif(TNL_USE_OPENMP)
        set(_thrust_host_device OMP)
    else()
        set(_thrust_host_device CPP)
    endif()
    thrust_create_target(ThrustHost HOST ${_thrust_host_device} DEVICE ${_thrust_host_device})
    thrust_create_target(ThrustCuda HOST CPP DEVICE CUDA) # NOTE: OMP does not work with CUDA

    foreach(target IN ITEMS ${BENCHMARKS})
        target_compile_definitions(${target} PUBLIC -DHAVE_THRUST)
        # HACK: Thrust adds its include directory as non-system directory, which leads
        # to warnings (and thus errors with TNL_USE_CI_FLAGS=ON) due to -Wextra-semi:
        # https://github.com/NVIDIA/cccl/issues/826
        target_include_directories(${target} SYSTEM PUBLIC ${_THRUST_INCLUDE_DIR})
    endforeach()
    if(TNL_BUILD_CPP_TARGETS)
        target_link_libraries(tnl-benchmark-blas PUBLIC ThrustHost)
    endif()
    if(TNL_BUILD_CUDA)
        target_link_libraries(tnl-benchmark-blas-cuda PUBLIC ThrustCuda)
    endif()
endif()
