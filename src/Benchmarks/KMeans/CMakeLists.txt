if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-kmeans-cuda tnl-benchmark-kmeans.cu)
    target_link_libraries(tnl-benchmark-kmeans-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-kmeans-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

if(TNL_BUILD_HIP)
    add_executable(tnl-benchmark-kmeans-hip tnl-benchmark-kmeans.hip)
    target_link_libraries(tnl-benchmark-kmeans-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-kmeans-hip RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-kmeans tnl-benchmark-kmeans.cpp)
    target_link_libraries(tnl-benchmark-kmeans PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-kmeans RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
