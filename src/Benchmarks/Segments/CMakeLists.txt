if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-segments-cuda tnl-benchmark-segments.cu)
    target_link_libraries(tnl-benchmark-segments-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-segments-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

if(TNL_BUILD_HIP)
    add_executable(tnl-benchmark-segments-hip tnl-benchmark-segments.hip)
    target_link_libraries(tnl-benchmark-segments-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-segments-hip RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-segments tnl-benchmark-segments.cpp)
    target_link_libraries(tnl-benchmark-segments PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-segments RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
