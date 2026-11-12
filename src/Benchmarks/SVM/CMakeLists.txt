if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-svm-cuda tnl-benchmark-svm.cu)
    target_link_libraries(tnl-benchmark-svm-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-svm-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

if(TNL_BUILD_HIP)
    add_executable(tnl-benchmark-svm-hip tnl-benchmark-svm.hip)
    target_link_libraries(tnl-benchmark-svm-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-svm-hip RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-svm tnl-benchmark-svm.cpp)
    target_link_libraries(tnl-benchmark-svm PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-svm RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
