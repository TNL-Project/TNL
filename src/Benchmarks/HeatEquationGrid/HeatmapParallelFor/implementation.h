
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
   const Real hx_inv = 1. / (hx * hx);
   const Real hy_inv = 1. / (hy * hy);

   auto timestep = params.timeStep ? params.timeStep : std::min(hx * hx, hy * hy);

   auto uxView = ux.getView(), auxView = aux.getView();

   auto xSize = params.xSize;
   auto xDomainSize = params.xDomainSize;
   auto yDomainSize = params.yDomainSize;
   auto sigma = params.sigma;

   auto init = [=] __cuda_callable__(int i, int j) mutable
   {
      auto index = j * xSize + i;

      auto x = i * hx - xDomainSize / 2.;
      auto y = j * hy - yDomainSize / 2.;

      uxView[index] = exp(sigma * (x * x + y * y));
   };

   TNL::Algorithms::ParallelFor2D<Device>::exec(1, 1, params.xSize - 1, params.ySize - 1, init);

   if (!writeGNUPlot("data.txt", params, ux))
      return false;

   auto next = [=] __cuda_callable__(int i, int j) mutable
   {
      auto index = j * xSize + i;
      auto center = 2 * uxView[index];

      auxView[index] = (uxView[index - 1] - center + uxView[index + 1]) * hx_inv +
                       (uxView[index - xSize] - center + uxView[index + xSize]) * hy_inv;
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
