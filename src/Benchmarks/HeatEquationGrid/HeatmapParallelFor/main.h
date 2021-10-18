
#include <iostream>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Containers/Array.h>
#include <TNL/Timer.h>
#include <TNL/Algorithms/ParallelFor.h>

#pragma once

class HeatmapSolver {
  public:
    class Parameters {
      public:
        const int xSize, ySize;
        const double xDomainSize, yDomainSize;
        const double sigma;
        const double timeStep, finalTime;
        const bool verbose;

        Parameters(const TNL::Config::ParameterContainer& parameters);

        static TNL::Config::ConfigDescription makeInputConfig();
      private:
    };

    bool solve(const Parameters &parameters, TNL::Timer &timer) const;

  private:
};

int main(int argc, char * argv[]) {
  TNL::Timer timer;
  auto config = HeatmapSolver::Parameters::makeInputConfig();

  TNL::Config::ParameterContainer parameters;
  if (!parseCommandLine(argc, argv, config, parameters))
    return EXIT_FAILURE;

  auto params = HeatmapSolver::Parameters(parameters);

  return 0;
}
