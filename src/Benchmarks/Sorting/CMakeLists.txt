if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-sort-cuda tnl-benchmark-sort.cu)
    target_link_libraries(tnl-benchmark-sort-cuda PUBLIC TNL::TNL)

    # Add include directories from CUDA samples for reference algorithms
    if(CUDA_SAMPLES_FOUND)
        target_include_directories(tnl-benchmark-sort-cuda SYSTEM PRIVATE ${CUDA_SAMPLES_INCLUDE_DIRS})
        target_compile_definitions(tnl-benchmark-sort-cuda PRIVATE HAVE_CUDA_SAMPLES)
    endif()

    find_package(CCCL REQUIRED)
    target_link_libraries(tnl-benchmark-sort-cuda PUBLIC CCCL::CCCL)

    install(TARGETS tnl-benchmark-sort-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-sort tnl-benchmark-sort.cpp)
    target_link_libraries(tnl-benchmark-sort PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-sort RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
