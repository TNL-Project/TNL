if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-ode-solvers-cuda tnl-benchmark-ode-solvers.cu)
    target_link_libraries(tnl-benchmark-ode-solvers-cuda PUBLIC TNL::TNL_CUDA)
    install(TARGETS tnl-benchmark-ode-solvers-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-ode-solvers tnl-benchmark-ode-solvers.cpp)
    target_link_libraries(tnl-benchmark-ode-solvers PUBLIC TNL::TNL_CXX)
    install(TARGETS tnl-benchmark-ode-solvers RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()
