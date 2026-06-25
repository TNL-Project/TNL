find_package(Ginkgo)
find_package(STRUMPACK)

# find_package(Trilinos) # This leads to wrong parameters for linker https://github.com/trilinos/Trilinos/issues/13396
find_package(Kokkos)
if(NOT TARGET SuiteSparse::UMFPACK)
    find_package(UMFPACK GLOBAL)
endif()

if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-linear-solvers-cuda tnl-benchmark-linear-solvers.cu)
    target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC TNL::TNL)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC CUDA::cusparse CUDA::cusolver)
    install(TARGETS tnl-benchmark-linear-solvers-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)

    find_package(cudss)
    if(cudss_FOUND)
        # Verify that cuDSS is compatible with the installed CUDA version
        # (cuDSS may have been built against a different CUDA major version)
        set(_cudss_compatible TRUE)
        if(DEFINED CUDAToolkit_VERSION_MAJOR)
            get_target_property(_cudss_location cudss IMPORTED_LOCATION_RELEASE)
            if(NOT _cudss_location)
                get_target_property(_cudss_location cudss IMPORTED_LOCATION)
            endif()
            if(_cudss_location AND EXISTS "${_cudss_location}")
                execute_process(
                    COMMAND readelf -d "${_cudss_location}"
                    OUTPUT_VARIABLE _cudss_readelf_output
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )
                string(REGEX MATCH "libcublas\\.so\\.([0-9]+)" _cudss_cublas_match "${_cudss_readelf_output}")
                if(_cudss_cublas_match)
                    set(_cudss_cublas_major "${CMAKE_MATCH_1}")
                    if(NOT "${_cudss_cublas_major}" STREQUAL "${CUDAToolkit_VERSION_MAJOR}")
                        message(STATUS "cuDSS was built against CUDA ${_cudss_cublas_major} but CUDA ${CUDAToolkit_VERSION_MAJOR} is installed, disabling cuDSS")
                        set(_cudss_compatible FALSE)
                    endif()
                endif()
            endif()
        endif()
        if(_cudss_compatible)
            target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_CUDSS")
            target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC cudss)
        endif()
        unset(_cudss_compatible)
        unset(_cudss_location)
        unset(_cudss_readelf_output)
        unset(_cudss_cublas_match)
        unset(_cudss_cublas_major)
    endif()

    if(TNL_BUILD_MPI)
        # enable MPI support in TNL
        target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_MPI")

        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC MPI::MPI_CXX)

        # FIXME: we can't link to ginkgo when using LLVM's libc++: https://github.com/ginkgo-project/ginkgo/issues/1571
        if(Ginkgo_FOUND AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_GINKGO")
            target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC Ginkgo::ginkgo)
        endif()

        if(STRUMPACK_FOUND)
            target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_STRUMPACK")
            target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC STRUMPACK::strumpack)
        endif()
    endif()

    if(TARGET SuiteSparse::UMFPACK)
        target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_UMFPACK")
        target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC SuiteSparse::UMFPACK)
        # reset SuiteSparse::UMFPACK to system target (upstream sets 0)
        set_property(TARGET SuiteSparse::UMFPACK PROPERTY SYSTEM 1)
    endif()

    if(Trilinos_FOUND AND Kokkos_ENABLE_CUDA)
        target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_TRILINOS -DKOKKOS_ARCH_ADA89")
        target_include_directories(tnl-benchmark-linear-solvers-cuda PUBLIC ${Trilinos_INCLUDE_DIRS})
        target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC ${Trilinos_LIBRARIES})
    endif()
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-linear-solvers tnl-benchmark-linear-solvers.cpp)
    target_link_libraries(tnl-benchmark-linear-solvers PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-linear-solvers RUNTIME DESTINATION bin COMPONENT benchmarks)

    if(TNL_BUILD_MPI)
        # enable MPI support in TNL
        target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_MPI")

        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(tnl-benchmark-linear-solvers PUBLIC MPI::MPI_CXX)

        # FIXME: we can't link to ginkgo when using LLVM's libc++: https://github.com/ginkgo-project/ginkgo/issues/1571
        if(Ginkgo_FOUND AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_GINKGO")
            target_link_libraries(tnl-benchmark-linear-solvers PUBLIC Ginkgo::ginkgo)
        endif()

        if(STRUMPACK_FOUND)
            target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_STRUMPACK")
            target_link_libraries(tnl-benchmark-linear-solvers PUBLIC STRUMPACK::strumpack)
        endif()
    endif()

    if(TARGET SuiteSparse::UMFPACK)
        target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_UMFPACK")
        target_link_libraries(tnl-benchmark-linear-solvers PUBLIC SuiteSparse::UMFPACK)
        # reset SuiteSparse::UMFPACK to system target (upstream sets 0)
        set_property(TARGET SuiteSparse::UMFPACK PROPERTY SYSTEM 1)
    endif()

    if(Trilinos_FOUND)
        target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_TRILINOS")
        target_include_directories(tnl-benchmark-linear-solvers PUBLIC ${Trilinos_INCLUDE_DIRS})
        target_link_libraries(tnl-benchmark-linear-solvers PUBLIC ${Trilinos_LIBRARIES})
    endif()
endif()
