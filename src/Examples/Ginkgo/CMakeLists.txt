# workaround for https://github.com/ginkgo-project/ginkgo/issues/1096
if(NOT Ginkgo_FOUND)
    find_package(Ginkgo QUIET)
endif()

set(CPP_EXAMPLES tnl-ginkgo-converted-ellpack tnl-ginkgo-wrapped-csr tnl-ginkgo-wrapped-operator tnl-ginkgo-PoissonEquation2D
                 tnl-ginkgo-PoissonEquation3D
)
set(CUDA_EXAMPLES tnl-ginkgo-converted-ellpack-cuda tnl-ginkgo-wrapped-csr-cuda tnl-ginkgo-wrapped-operator-cuda
                  tnl-ginkgo-PoissonEquation2D-cuda tnl-ginkgo-PoissonEquation3D-cuda
)

# FIXME: we can't link to ginkgo when using LLVM's libc++: https://github.com/ginkgo-project/ginkgo/issues/1571
if(Ginkgo_FOUND AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    foreach(target IN ITEMS ${CPP_EXAMPLES})
        add_executable(${target} ${target}.cpp)
        target_link_libraries(${target} TNL::TNL Ginkgo::ginkgo)
        install(TARGETS ${target} RUNTIME DESTINATION bin COMPONENT examples)
    endforeach()
    # FIXME: we can't link to ginkgo when using LLVM's libc++: https://github.com/ginkgo-project/ginkgo/issues/1571
    if(TNL_BUILD_CUDA AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        foreach(target IN ITEMS ${CUDA_EXAMPLES})
            add_executable(${target} ${target}.cu)
            target_link_libraries(${target} TNL::TNL Ginkgo::ginkgo)
            install(TARGETS ${target} RUNTIME DESTINATION bin COMPONENT examples)
        endforeach()
    endif()
endif()
