
#pragma once

#include "GridBenchmark.h"

#define GRID_DIMENSION 2

int main(int argc, char* argv[]) {
   GridBenchmark benchmark;

   auto config = GridBenchmark::makeInputConfig(GRID_DIMENSION);

   TNL::Config::ParameterContainer parameters;

   if (!parseCommandLine(argc, argv, config, parameters))
      return EXIT_FAILURE;

   return benchmark.runBenchmark<GRID_DIMENSION>(parameters);
}
