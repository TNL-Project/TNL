
#pragma once

#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <array>

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Containers/Array.h>

template <typename Real>
class HeatmapSolver
{
public:
   class Parameters
   {
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
                     const Parameters &parameters,
                     const TNL::Containers::Array<Real, Device> &map) const;
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

template <typename Real>
HeatmapSolver<Real>::Parameters::Parameters(const TNL::Config::ParameterContainer &parameters) : xSize(parameters.getParameter<int>("grid-x-size")),
                                                                                                 ySize(parameters.getParameter<int>("grid-y-size")),
                                                                                                 xDomainSize(parameters.getParameter<Real>("domain-x-size")),
                                                                                                 yDomainSize(parameters.getParameter<Real>("domain-y-size")),
                                                                                                 sigma(parameters.getParameter<Real>("sigma")),
                                                                                                 timeStep(parameters.getParameter<Real>("time-step")),
                                                                                                 finalTime(parameters.getParameter<Real>("final-time")),
                                                                                                 verbose(parameters.getParameter<bool>("verbose")) {}

template <typename Real>
template <typename Device>
bool HeatmapSolver<Real>::writeGNUPlot(const std::string &filename,
                                       const HeatmapSolver<Real>::Parameters &params,
                                       const TNL::Containers::Array<Real, Device> &map) const
{
   std::ofstream out(filename, std::ios::out);

   if (!out.is_open())
      return false;

   for (int j = 0; j < params.ySize; j++)
      for (int i = 0; i < params.xSize; i++)
         out << i << " " << j << " " << map[j * params.xSize + i] << std::endl;

   return out.good();
}
