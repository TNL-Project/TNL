# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-ndarray tnl-benchmark-ndarray.cpp)
    target_link_libraries(tnl-benchmark-ndarray PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-ndarray RUNTIME DESTINATION bin COMPONENT benchmarks)

    add_executable(tnl-benchmark-ndarray-boundary tnl-benchmark-ndarray-boundary.cpp)
    target_link_libraries(tnl-benchmark-ndarray-boundary PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-ndarray-boundary RUNTIME DESTINATION bin COMPONENT benchmarks)

    add_executable(tnl-benchmark-reduction tnl-benchmark-reduction.cpp)
    target_link_libraries(tnl-benchmark-reduction PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-reduction RUNTIME DESTINATION bin COMPONENT benchmarks)

    add_executable(tnl-benchmark-ndarray-reduction tnl-benchmark-ndarray-reduction.cpp)
    target_link_libraries(tnl-benchmark-ndarray-reduction PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-ndarray-reduction RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-ndarray-cuda tnl-benchmark-ndarray-cuda.cu)
    target_link_libraries(tnl-benchmark-ndarray-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-ndarray-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)

    add_executable(tnl-benchmark-ndarray-boundary-cuda tnl-benchmark-ndarray-boundary-cuda.cu)
    target_link_libraries(tnl-benchmark-ndarray-boundary-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-ndarray-boundary-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)

    add_executable(tnl-benchmark-reduction-cuda tnl-benchmark-reduction-cuda.cu)
    target_link_libraries(tnl-benchmark-reduction-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-reduction-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)

    add_executable(tnl-benchmark-ndarray-reduction-cuda tnl-benchmark-ndarray-reduction-cuda.cu)
    target_link_libraries(tnl-benchmark-ndarray-reduction-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-ndarray-reduction-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

if(TNL_BUILD_HIP)
    add_executable(tnl-benchmark-ndarray-hip tnl-benchmark-ndarray-hip.hip)
    target_link_libraries(tnl-benchmark-ndarray-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-ndarray-hip RUNTIME DESTINATION bin COMPONENT benchmarks)

    add_executable(tnl-benchmark-ndarray-boundary-hip tnl-benchmark-ndarray-boundary-hip.hip)
    target_link_libraries(tnl-benchmark-ndarray-boundary-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-ndarray-boundary-hip RUNTIME DESTINATION bin COMPONENT benchmarks)

    add_executable(tnl-benchmark-reduction-hip tnl-benchmark-reduction-hip.hip)
    target_link_libraries(tnl-benchmark-reduction-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-reduction-hip RUNTIME DESTINATION bin COMPONENT benchmarks)

    add_executable(tnl-benchmark-ndarray-reduction-hip tnl-benchmark-ndarray-reduction-hip.hip)
    target_link_libraries(tnl-benchmark-ndarray-reduction-hip PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-ndarray-reduction-hip RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
