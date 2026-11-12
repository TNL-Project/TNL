if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-als-cuda tnl-benchmark-als.cu)
    target_link_libraries(tnl-benchmark-als-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-als-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

if(TNL_BUILD_HIP)
    add_executable(tnl-benchmark-als-hip tnl-benchmark-als.hip)
    target_link_libraries(tnl-benchmark-als-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-als-hip RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-als tnl-benchmark-als.cpp)
    target_link_libraries(tnl-benchmark-als PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-als RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
