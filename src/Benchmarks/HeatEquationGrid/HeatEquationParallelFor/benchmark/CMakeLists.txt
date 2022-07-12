if (BUILD_CUDA)
   CUDA_ADD_EXECUTABLE( tnl-benchmark-heat-equation-parallel-for ../implementation.h ../../Base/HeatEquationSolver.h ../../Base/HeatEquationSolverBenchmark.h main.h main.cu )
ELSE()
   add_executable( tnl-benchmark-heat-equation-parallel-for ../implementation.h ../../Base/HeatEquationSolver.h ../../Base/HeatEquationSolverBenchmark.h main.h main.cpp )
ENDIF()

install( TARGETS tnl-benchmark-heat-equation-parallel-for RUNTIME DESTINATION bin )
