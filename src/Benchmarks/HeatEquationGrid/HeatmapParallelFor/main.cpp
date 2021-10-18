#include "main.h"

TNL::Config::ConfigDescription HeatmapSolver::Parameters::makeInputConfig()
{
  TNL::Config::ConfigDescription config;

  config.addEntry<TNL::String>("device", "Device the computation will run on.", "host");
  config.addEntryEnum<TNL::String>("host");

#ifdef HAVE_CUDA
  config.addEntryEnum<TNL::String>("cuda");
#endif

  config.addEntry<int>("grid-x-size", "Grid size along x-axis.", 100);
  config.addEntry<int>("grid-y-size", "Grid size along y-axis.", 100);

  config.addEntry<double>("domain-x-size", "Domain size along x-axis.", 2.0);
  config.addEntry<double>("domain-y-size", "Domain size along y-axis.", 2.0);

  config.addEntry<double>("sigma", "Sigma in exponential initial condition.", 2.0);

  config.addEntry<double>("time-step", "Time step. By default it is proportional to one over space step square.", 0.0);
  config.addEntry<double>("final-time", "Final time of the simulation.", 1.0);
  config.addEntry<bool>("verbose", "Verbose mode.", true);

  return config;
}

HeatmapSolver::Parameters::Parameters(const TNL::Config::ParameterContainer& parameters):
  xSize(parameters.getParameter<int>("grid-x-size")),
  ySize(parameters.getParameter<int>("grid-y-size")),
  xDomainSize(parameters.getParameter<double>("domain-x-size")),
  yDomainSize(parameters.getParameter<double>("domain-y-size")),
  sigma(parameters.getParameter<double>("sigma")),
  timeStep(parameters.getParameter<double>("time-step")),
  finalTime(parameters.getParameter<double>("final-time")),
  verbose(parameters.getParameter<bool>("verbose")) {}

bool HeatmapSolver::Solve(const HeatmapSolver::Parameters& params, TNL::Timer& timer) const {
  // This is always an external storage for grid.
  TNL::Container::Array ux(params.xSize * params.ySize),  // data at step u
                        aux(params.xSize * params.ySize); // data at step u + 1

  // Invalidate ux/aux
  ux = 0;
  aux = 0;

  const double hx = params.xDomainSize / (double)params.xSize;
  const double hy = params.yDomainSize / (double)params.ySize;

  auto uxView = ux.getView(), auxView = aux.getView();

  timer.reset();

  // TODO: - Initial Condition


  auto horizontalBoundaryCondition = [=] __cuda_callable__ (int i) {
    aux[i] = 0;
    aux[(params.ySize - 1) * params.xSize + i] = 0;
  };

  auto verticalBoundaryCondition = [=] __cuda_callable__(int i) {
    aux[j * params.ySize] = 0;
    aux[j * params.ySize + params.xSize - 1] = 0;
  };

  auto next = [=] __cuda_callable__(int i, int j) {
    auto index = j * params.ySize + i;

    aux[index] = (u[c - 1] - 2 * u[c] + u[c + 1]) / hx +
                 (u[c - params.xSize] - 2 * u[c] + u[c + params.xSize]) / hy;
  };

  double time = 0;

  while (time < params.finalTime) {
    // TODO: - Do we really need this
    TNL::Algorithm::ParallelFor(0, params.xSize, horizontalBoundaryCondition);
    TNL::Algorithm::ParallelFor(0, params.ySize, verticalBoundaryCondition);
    TNL::Algorithm::ParallelFor2D(1, 1, params.xSize - 1, params.ySize - 1, next);

    time += params.timeStep;
  }

  return false;
}
