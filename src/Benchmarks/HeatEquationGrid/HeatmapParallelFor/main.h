
#include <iostream>
#include <fstream>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/ParallelFor.h>

#include "../Base/HeatmapSolver.h"

#pragma once

/***
 * Grid parameters:
 *
 * ySize-1|j                                (ySize - 1) * xSize + xSize - 1
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
 *  0                                                xSize-1|i
 *
 *  j * xSize + i
 ***/
template <typename Real>
template <typename Device>
bool HeatmapSolver<Real>::solve(const HeatmapSolver<Real>::Parameters &params) const
{
   TNL::Containers::Array<Real, Device> ux(params.xSize * params.ySize), // data at step u
                                        aux(params.xSize * params.ySize);// data at step u + 1

   // Invalidate ux/aux
   ux = 0;
   aux = 0;

   const Real hx = params.xDomainSize / (Real)params.xSize;
   const Real hy = params.yDomainSize / (Real)params.ySize;
   const Real hx_inv = 1 / (hx * hx);
   const Real hy_inv = 1 / (hy * hy);

   auto timestep = params.timeStep ? params.timeStep : std::min(hx * hx, hy * hy);

   auto uxView = ux.getView(), auxView = aux.getView();

   auto init = [=] __cuda_callable__(int i, int j) mutable
   {
      auto index = j * params.xSize + i;

      auto x = i * hx - params.xDomainSize / 2;
      auto y = j * hy - params.yDomainSize / 2;

      uxView[index] = exp(params.sigma * (x * x + y * y));
   };

   TNL::Algorithms::ParallelFor2D<Device>::exec(1, 1, params.xSize - 1, params.ySize - 1, init);

   if (!writeGNUPlot("data.txt", params, ux))
      return false;

   auto next = [=] __cuda_callable__(int i, int j) mutable
   {
      auto index = j * params.ySize + i;

      auxView[index] = (uxView[index - 1] - 2 * uxView[index] + uxView[index + 1]) * hx_inv +
                       (uxView[index - params.xSize] - 2 * uxView[index] + uxView[index + params.xSize]) * hy_inv;
   };

   auto update = [=] __cuda_callable__(int i) mutable
   {
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
