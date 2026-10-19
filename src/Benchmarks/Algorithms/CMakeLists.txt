# the benchmark tunes the GPU scan kernels, so there is no host-only executable
if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-scan-cuda tnl-benchmark-scan.cu)
    target_link_libraries(tnl-benchmark-scan-cuda PUBLIC TNL::TNL)
    find_package(CCCL REQUIRED)
    target_link_libraries(tnl-benchmark-scan-cuda PUBLIC CCCL::CCCL)
    install(TARGETS tnl-benchmark-scan-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

if(TNL_BUILD_HIP)
    add_executable(tnl-benchmark-scan-hip tnl-benchmark-scan.hip)
    target_link_libraries(tnl-benchmark-scan-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-scan-hip RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
