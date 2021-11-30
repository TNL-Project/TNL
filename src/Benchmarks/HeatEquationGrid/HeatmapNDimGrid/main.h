
#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <array>
#include <bitset>

#include "../Base/HeatmapSolver.h"
#include "implementation.h"

int main(int argc, char* argv[]) {
   using Real = double;

   auto config = HeatmapSolver<Real>::Parameters::makeInputConfig();

   TNL::Config::ParameterContainer parameters;
   if (!parseCommandLine(argc, argv, config, parameters))
      return EXIT_FAILURE;

   auto device = parameters.getParameter<TNL::String>("device");
   auto params = HeatmapSolver<Real>::Parameters(parameters);

   HeatmapSolver<Real> solver;

   if (device == "host" && !solver.solve<TNL::Devices::Host>(params))
      return EXIT_FAILURE;

#ifdef HAVE_CUDA
   if (device == "cuda" && !solver.solve<TNL::Devices::Cuda>(params))
      return EXIT_FAILURE;
#endif

   return EXIT_SUCCESS;
}


/*
int main(int argc, char *argv[]) {
   Grid<2, int, TNL::Devices::Cuda> grid;

   grid.setDimensions(5, 5);

   auto fn_entity = [=] __cuda_callable__ (GridEntity<2, int> entity) {
      printf("%d %d %d\n", entity.getCoordinates()[0], entity.getCoordinates()[1], entity.getIndex());
   };

   Container<2, int> direction { 0, 0 };

   grid.traverse({ 1, 1 }, { 4, 4 }, { direction }, fn_entity);

   return 0;
}
*/
