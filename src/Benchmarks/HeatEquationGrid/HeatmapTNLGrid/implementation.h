#pragma once

#include "TNL/Meshes/NDimGrid.h"
#include "TNL/Meshes/GridDetails/Grid2D.h"
#include "../Base/HeatmapSolver.h"

template <typename Real>
template <typename Device>
bool HeatmapSolver<Real>::solve(const HeatmapSolver<Real>::Parameters& params) const {
   using Grid2D = TNL::Meshes::Grid<2, Real, Device, int>;

   Grid2D grid;

   // Grid implementation defines its dimensions in the amount of edges.
   // To align it size to all other benchmarks substract 1
   grid.setDimensions(params.xSize - 1, params.ySize - 1);

   // TODO: - Improve style of access. It is counterintuitive for person, who doesn't know C++ well
   auto verticesCount = grid.template getEntitiesCount<0>();

   TNL::Containers::Array<Real, Device> ux(verticesCount), // data at step u
                                        aux(verticesCount);// data at step u + 1

   // Invalidate ux/aux
   ux = 0;
   aux = 0;

   const Real hx = params.xDomainSize / (Real)params.xSize;
   const Real hy = params.yDomainSize / (Real)params.ySize;
   const Real hx_inv = 1. / (hx * hx);
   const Real hy_inv = 1. / (hy * hy);

   auto timestep = params.timeStep ? params.timeStep : std::min(hx * hx, hy * hy);

   auto uxView = ux.getView(),
        auxView = aux.getView();

   auto xDomainSize = params.xDomainSize;
   auto yDomainSize = params.yDomainSize;
   auto sigma = params.sigma;

   auto init = [=] __cuda_callable__(const typename Grid2D::EntityType<0> &entity) mutable {
      auto index = entity.getIndex();

      auto x = entity.getCoordinates().x() * hx - xDomainSize / 2.;
      auto y = entity.getCoordinates().y() * hy - yDomainSize / 2.;

      uxView[index] = exp(sigma * (x * x + y * y));
   };

   grid.template forInterior<0>(init);

   if (!writeGNUPlot("data.txt", params, ux))
      return false;

   auto width = grid.getDimensions().x() + 1;
   auto next = [=] __cuda_callable__(const typename Grid2D::EntityType<0>&entity) mutable {
      auto index = entity.getIndex();
      auto center = 2 * uxView[index];

      auxView[index] = (uxView[index - 1] - center + uxView[index + 1]) * hx_inv +
                        (uxView[index - width] - center + uxView[index + width]) * hy_inv;
   };

   auto update = [=] __cuda_callable__(const typename Grid2D::EntityType<0>&entity) mutable {
      auto index = entity.getIndex();

      uxView[index] += auxView[index] * timestep;
   };

   Real start = 0;

   while (start < params.finalTime) {
      grid.template forInterior<0>(next);
      grid.template forInterior<0>(update);

      start += timestep;
   }

   return writeGNUPlot("data_final.txt", params, ux);

   return false;
}
