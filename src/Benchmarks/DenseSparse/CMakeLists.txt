if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-dense-sparse-cuda tnl-benchmark-dense-sparse.cu)
    target_link_libraries(tnl-benchmark-dense-sparse-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-dense-sparse-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

if(TNL_BUILD_HIP)
    add_executable(tnl-benchmark-dense-sparse-hip tnl-benchmark-dense-sparse.hip)
    target_link_libraries(tnl-benchmark-dense-sparse-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-dense-sparse-hip RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-dense-sparse tnl-benchmark-dense-sparse.cpp)
    target_link_libraries(tnl-benchmark-dense-sparse PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-dense-sparse RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
