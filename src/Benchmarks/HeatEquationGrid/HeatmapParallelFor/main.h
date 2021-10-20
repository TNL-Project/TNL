
#include<iostream>
#include<fstream>
#include<TNL/Config/parseCommandLine.h>
#include<TNL/Containers/Array.h>
#include<TNL/Algorithms/ParallelFor.h>

#pragma once

template<typename Real>
class HeatmapSolver {
  public:
    class Parameters {
      public:
        const int xSize, ySize;
        const Real xDomainSize, yDomainSize;
        const Real sigma;
        const Real timeStep, finalTime;
        const bool verbose;

        Parameters(const TNL::Config::ParameterContainer &parameters);

        static TNL::Config::ConfigDescription makeInputConfig();
    };

    template <typename Device>
    bool solve(const Parameters &parameters) const;

  private:
    template <typename Device>
    bool writeGNUPlot(const std::string &filename,
                      const Parameters& parameters,
                      const TNL::Containers::Array<Real, Device>& map) const;
};

template <typename Real>
TNL::Config::ConfigDescription HeatmapSolver<Real>::Parameters::makeInputConfig()
{
  TNL::Config::ConfigDescription config;

  config.addEntry<TNL::String>("device", "Device the computation will run on.", "host");
  config.addEntryEnum<TNL::String>("host");

#ifdef HAVE_CUDA
  config.addEntryEnum<TNL::String>("cuda");
#endif

  config.addEntry<int>("grid-x-size", "Grid size along x-axis.", 100);
  config.addEntry<int>("grid-y-size", "Grid size along y-axis.", 100);

  config.addEntry<Real>("domain-x-size", "Domain size along x-axis.", 2.0);
  config.addEntry<Real>("domain-y-size", "Domain size along y-axis.", 2.0);

  config.addEntry<Real>("sigma", "Sigma in exponential initial condition.", 1.0);

  config.addEntry<Real>("time-step", "Time step. By default it is proportional to one over space step square.", 0.0);
  config.addEntry<Real>("final-time", "Final time of the simulation.", 0.012);
  config.addEntry<bool>("verbose", "Verbose mode.", true);

  return config;
}

template<typename Real>
HeatmapSolver<Real>::Parameters::Parameters(const TNL::Config::ParameterContainer &parameters) : xSize(parameters.getParameter<int>("grid-x-size")),
                                                                                                 ySize(parameters.getParameter<int>("grid-y-size")),
                                                                                                 xDomainSize(parameters.getParameter<Real>("domain-x-size")),
                                                                                                 yDomainSize(parameters.getParameter<Real>("domain-y-size")),
                                                                                                 sigma(parameters.getParameter<Real>("sigma")),
                                                                                                 timeStep(parameters.getParameter<Real>("time-step")),
                                                                                                 finalTime(parameters.getParameter<Real>("final-time")),
                                                                                                 verbose(parameters.getParameter<bool>("verbose")) {}

/***
 * Grid parameters:
 *
 * ySize|j                                          (ySize - 1) * xSize + xSize - 1
 *  |------------------------------------------------------
 *  |
 *  |
 *  |
 *  |
 *  |
 *  |
 *  |
 *  |
 *  |
 *  |------------------------------------------------------>
 *
 *  0                                                xSize|i
 *
 *  j * xSize + i
 ***/
template<typename Real>
template<typename Device>
bool HeatmapSolver<Real>::solve(const HeatmapSolver<Real>::Parameters &params) const
{
  // This is always an external storage for grid.
  TNL::Containers::Array<Real, Device> ux(params.xSize * params.ySize),  // data at step u
                                       aux(params.xSize * params.ySize); // data at step u + 1

  // Invalidate ux/aux
  ux = 0;
  aux = 0;

  const Real hx = params.xDomainSize / (Real)params.xSize;
  const Real hy = params.yDomainSize / (Real)params.ySize;
  const Real hx_inv = 1 / (hx * hx);
  const Real hy_inv = 1 / (hy * hy);

  auto timestep = params.timeStep ? params.timeStep : std::min(hx * hx, hy * hy);

  auto uxView = ux.getView(), auxView = aux.getView();

  auto init = [=] __cuda_callable__(int i, int j) mutable {
    auto index = j * params.xSize + i;

    auto x = i * hx - params.xDomainSize / 2;
    auto y = j * hy - params.yDomainSize / 2;

    uxView[index] = exp(params.sigma * (x * x + y * y));
  };

  TNL::Algorithms::ParallelFor2D<Device>::exec(1, 1, params.xSize - 1, params.ySize - 1, init);

  if (!writeGNUPlot("data.txt", params, ux))
    return false;

  auto next = [=] __cuda_callable__(int i, int j) mutable {
    auto index = j * params.ySize + i;

    auxView[index] = (uxView[index - 1] - 2 * uxView[index] + uxView[index + 1]) * hx_inv +
                     (uxView[index - params.xSize] - 2 * uxView[index] + uxView[index + params.xSize]) * hy_inv;
  };

  auto update = [=] __cuda_callable__(int i) mutable {
    uxView[i] += auxView[i] * timestep;
  };

  Real start = 0;

  while (start < params.finalTime)
  {
    TNL::Algorithms::ParallelFor2D<Device>::exec(1, 1, params.xSize - 1, params.ySize - 1, next);
    TNL::Algorithms::ParallelFor<Device>::exec(0, params.xSize * params.ySize, update);

    start += timestep;
  }

  return writeGNUPlot("data_final.txt", params, ux);
}

template <typename Real>
template <typename Device>
bool HeatmapSolver<Real>::writeGNUPlot(const std::string &filename,
                                       const HeatmapSolver<Real>::Parameters &params,
                                       const TNL::Containers::Array<Real, Device> &map) const {
  std::ofstream out(filename, std::ios::out);

  if (!out.is_open())
    return false;

  for (int j = 0; j < params.ySize; j++)
    for (int i = 0; i < params.xSize; i++)
      out << i << " " << j << " " << map[j * params.xSize + i] << std::endl;

  return out.good();
}

int main(int argc, char * argv[]) {
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
