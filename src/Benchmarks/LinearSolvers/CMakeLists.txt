find_package(Ginkgo)
find_package(STRUMPACK)

# find_package(Trilinos) # This leads to wrong parameters for linker https://github.com/trilinos/Trilinos/issues/13396
find_package(Kokkos)
if(NOT TARGET SuiteSparse::UMFPACK)
    find_package(UMFPACK GLOBAL)
endif()

if(TNL_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(cudss)
endif()

if(TNL_BUILD_HIP)
    find_package(hipsolver REQUIRED)
endif()

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-benchmark-linear-solvers
    SOURCES tnl-benchmark-linear-solvers.h
    BACKENDS HOST CUDA HIP
    OPTIONAL_HOST
    CUDA_LINK_LIBRARIES CUDA::cusparse CUDA::cusolver
    HIP_LINK_LIBRARIES roc::hipsolver
    OUTPUT_TARGETS targets
    OUTPUT_BACKENDS backends
)

foreach(target backend IN ZIP_LISTS targets backends)
    if(backend STREQUAL "CUDA" AND cudss_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_CUDSS")
        target_link_libraries(${target} PUBLIC cudss)
    endif()

    if(TNL_BUILD_MPI)
        # enable MPI support in TNL
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")

        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

        # FIXME: we can't link to ginkgo when using LLVM's libc++: https://github.com/ginkgo-project/ginkgo/issues/1571
        if(
            Ginkgo_FOUND
            AND (
                (backend STREQUAL "CUDA" AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
                OR (backend STREQUAL "HOST" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            )
        )
            target_compile_definitions(${target} PUBLIC "-DHAVE_GINKGO")
            target_link_libraries(${target} PUBLIC Ginkgo::ginkgo)
        endif()

        if(STRUMPACK_FOUND)
            target_compile_definitions(${target} PUBLIC "-DHAVE_STRUMPACK")
            target_link_libraries(${target} PUBLIC STRUMPACK::strumpack)
        endif()
    endif()

    if(TARGET SuiteSparse::UMFPACK)
        target_compile_definitions(${target} PUBLIC "-DHAVE_UMFPACK")
        target_link_libraries(${target} PUBLIC SuiteSparse::UMFPACK)
    endif()

    if(backend STREQUAL "CUDA" AND Trilinos_FOUND AND Kokkos_ENABLE_CUDA)
        target_compile_definitions(${target} PUBLIC "-DHAVE_TRILINOS -DKOKKOS_ARCH_ADA89")
        target_include_directories(${target} PUBLIC ${Trilinos_INCLUDE_DIRS})
        target_link_libraries(${target} PUBLIC ${Trilinos_LIBRARIES})
    elseif(backend STREQUAL "HOST" AND Trilinos_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_TRILINOS")
        target_include_directories(${target} PUBLIC ${Trilinos_INCLUDE_DIRS})
        target_link_libraries(${target} PUBLIC ${Trilinos_LIBRARIES})
    endif()
endforeach()

if(TARGET SuiteSparse::UMFPACK)
    # reset SuiteSparse::UMFPACK to system target (upstream sets 0)
    set_property(TARGET SuiteSparse::UMFPACK PROPERTY SYSTEM 1)
endif()

if(targets)
    install(TARGETS ${targets} RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
