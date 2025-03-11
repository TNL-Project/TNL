
find_package (Eigen3 )
find_package(ortools)
find_package(GUROBI)

if(TNL_BUILD_CUDA)
    add_executable(tnl-benchmark-lp-solvers-cuda tnl-benchmark-lp-solvers.cu)
    target_link_libraries(tnl-benchmark-lp-solvers-cuda PUBLIC TNL::TNL_CUDA)
    install(TARGETS tnl-benchmark-lp-solvers-cuda RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-benchmark-lp-solvers tnl-benchmark-lp-solvers.cpp)
    target_link_libraries(tnl-benchmark-lp-solvers PUBLIC TNL::TNL_CXX)
    install(TARGETS tnl-benchmark-lp-solvers RUNTIME DESTINATION bin COMPONENT benchmarks)

    if(GUROBI_FOUND)
        target_compile_definitions(tnl-benchmark-lp-solvers PUBLIC "-DHAVE_GUROBI")
        target_include_directories(tnl-benchmark-lp-solvers PUBLIC ${GUROBI_INCLUDE_DIRS})
        target_link_libraries(tnl-benchmark-lp-solvers PUBLIC ${GUROBI_LIBRARIES})
    endif()
# if(ortools_FOUND AND Eigen3_FOUND)
# target_compile_definitions(tnl-benchmark-lp-solvers PUBLIC "-DHAVE_ORTOOLS")
# target_link_libraries(tnl-benchmark-lp-solvers PUBLIC TNL::TNL_CXX ortools::ortools Eigen3::Eigen)

# add_executable(pdlp-solve pdlp_solve.cc)
# target_include_directories(pdlp-solve PUBLIC ${EIGEN3_INCLUDE_DIR})
# target_link_libraries(pdlp-solve PUBLIC ortools::ortools Eigen3::Eigen)
# endif()
endif()
