find_package(Ginkgo)

set(EXAMPLES
    tnl-ginkgo-converted-ellpack
    tnl-ginkgo-wrapped-csr
    tnl-ginkgo-wrapped-operator
    tnl-ginkgo-PoissonEquation2D
    tnl-ginkgo-PoissonEquation3D
)

# FIXME: we can't link to ginkgo when using LLVM's libc++: https://github.com/ginkgo-project/ginkgo/issues/1571
if(Ginkgo_FOUND AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(GINKGO_BACKENDS HOST)
    # FIXME: we can't link to ginkgo when using LLVM's libc++: https://github.com/ginkgo-project/ginkgo/issues/1571
    if(TNL_BUILD_CUDA AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        list(APPEND GINKGO_BACKENDS CUDA)
    endif()
    foreach(example IN ITEMS ${EXAMPLES})
        tnl_add_backend_executables(
            ${example}
            SOURCES ${example}.cpp
            BACKENDS ${GINKGO_BACKENDS}
            LINK_LIBRARIES Ginkgo::ginkgo
            OUTPUT_TARGETS targets
        )
        install(TARGETS ${targets} RUNTIME DESTINATION bin COMPONENT examples)
    endforeach()
endif()
