find_package(STRUMPACK)
find_package(Trilinos)
find_package(Kokkos)
find_package(UMFPACK)

if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-linear-solvers-cuda tnl-benchmark-linear-solvers.cu)
    target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC TNL::TNL)
    target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_CUSPARSE")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC CUDA::cusparse)
    install(TARGETS tnl-benchmark-linear-solvers-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)

    if(TNL_BUILD_MPI)

        # enable MPI support in TNL
        target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_MPI")

        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC MPI::MPI_CXX)
        if(Ginkgo_FOUND)
            target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_GINKGO")
            target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC Ginkgo::ginkgo)
        endif()
    endif()

    if(UMFPACK_FOUND)
        target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_UMFPACK")
        target_include_directories(tnl-benchmark-linear-solvers-cuda PUBLIC ${UMFPACK_INCLUDE_DIR})
        target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC -lumfpack)
    endif()

    if(STRUMPACK_FOUND)
        target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_STRUMPACK")
        target_include_directories(tnl-benchmark-linear-solvers-cuda PUBLIC ${STRUMPACK_INCLUDE_DIR})
        target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC STRUMPACK::strumpack)
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
        if(Ginkgo_FOUND)
            target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_GINKGO")
            target_link_libraries(tnl-benchmark-linear-solvers PUBLIC TNL::TNL_CXX Ginkgo::ginkgo)
        endif()
    endif()

    if(UMFPACK_FOUND)
        target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_UMFPACK")
        target_include_directories(tnl-benchmark-linear-solvers PUBLIC ${UMFPACK_INCLUDE_DIR})
        target_link_libraries(tnl-benchmark-linear-solvers PUBLIC -lumfpack)
    endif()

    if(STRUMPACK_FOUND)
        target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_STRUMPACK")
        target_include_directories(tnl-benchmark-linear-solvers PUBLIC ${STRUMPACK_INCLUDE_DIR})
        target_link_libraries(tnl-benchmark-linear-solvers PUBLIC STRUMPACK::strumpack)
    endif()

    if(Trilinos_FOUND)
        target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_TRILINOS")
        target_include_directories(tnl-benchmark-linear-solvers PUBLIC ${Trilinos_INCLUDE_DIRS})
        target_link_libraries(tnl-benchmark-linear-solvers PUBLIC ${Trilinos_LIBRARIES})
    endif()

endif()

