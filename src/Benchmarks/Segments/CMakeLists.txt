if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-segments-cuda tnl-benchmark-segments.cu)
    target_link_libraries(tnl-benchmark-segments-cuda PUBLIC TNL::TNL)
    install(TARGETS tnl-benchmark-segments-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

add_executable(tnl-benchmark-segments tnl-benchmark-segments.cpp)
target_link_libraries(tnl-benchmark-segments PUBLIC TNL::TNL)
install(TARGETS tnl-benchmark-segments RUNTIME DESTINATION bin COMPONENT benchmarks)
