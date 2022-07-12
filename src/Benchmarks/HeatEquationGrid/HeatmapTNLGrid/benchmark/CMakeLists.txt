if (BUILD_CUDA)
   CUDA_ADD_EXECUTABLE( tnl-benchmark-heat-equation-grid ../implementation.h ../../Base/HeatmapSolver.h ../../Base/HeatmapSolverBenchmark.h main.h main.cu )
ELSE()
   add_executable( tnl-benchmark-heat-equation-grid ../implementation.h ../../Base/HeatmapSolver.h ../../Base/HeatmapSolverBenchmark.h main.h main.cpp )
ENDIF()

install( TARGETS tnl-benchmark-heat-equation-grid RUNTIME DESTINATION bin )
