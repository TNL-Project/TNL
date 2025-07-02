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
    target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_CUSPARSE")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC CUDA::cusparse)
    install(TARGETS tnl-benchmark-linear-solvers-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)

    find_package(cudss)
    if(cudss_FOUND)
        target_compile_definitions(tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_CUDSS")
        target_link_libraries(tnl-benchmark-linear-solvers-cuda PUBLIC cudss)
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
    endif()

    if(Trilinos_FOUND)
        target_compile_definitions(tnl-benchmark-linear-solvers PUBLIC "-DHAVE_TRILINOS")
        target_include_directories(tnl-benchmark-linear-solvers PUBLIC ${Trilinos_INCLUDE_DIRS})
        target_link_libraries(tnl-benchmark-linear-solvers PUBLIC ${Trilinos_LIBRARIES})
    endif()
endif()
