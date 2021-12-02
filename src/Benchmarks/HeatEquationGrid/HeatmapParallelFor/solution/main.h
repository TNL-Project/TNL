
#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <array>
#include <bitset>

#include "../../Base/HeatmapSolver.h"
#include "../implementation.h"

int main(int argc, char *argv[]) {
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
