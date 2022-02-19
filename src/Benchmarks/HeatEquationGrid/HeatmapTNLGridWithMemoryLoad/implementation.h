
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Containers/Array.h>
#include <TNL/Meshes/Grid.h>

#include "../Base/HeatmapSolver.h"

#pragma once

template<int Size, typename Index, typename Real>
class Entity;

template <size_t Value, size_t Power>
constexpr size_t pow() {
   size_t result = 1;

   for (size_t i = 0; i < Power; i++) {
      result *= Value;
   }

   return result;
}

constexpr int spaceStepsPowers = 5;

template<int Size, typename Index, typename Real>
class Grid {
   public:
      __cuda_callable__ inline
      Entity<Size, Index, Real> getEntity(Index i, Index j) const {
         Entity<Size, Index, Real> entity;

         entity.i = i;
         entity.j = j;

         return entity;
      }

      TNL::Containers::StaticVector<Size, Index> dimensions;
      TNL::Containers::StaticVector<1 << Size, Index> entitiesCountAlongBases;
      TNL::Containers::StaticVector<Size + 1, Index> cumulativeEntitiesCountAlongBases;

      TNL::Containers::StaticVector<Size, Real> origin, proportions, spaceSteps;

      TNL::Containers::StaticVector<std::integral_constant<Index, pow<spaceStepsPowers, Size>()>::value, Real> spaceProducts;
};

template<int Size, typename Index, typename Real>
class Entity {
   public:
      __cuda_callable__ inline
      Entity() {};

      // const Grid<Size, Index, Real>& grid;

      Index i, j;
      Index index;
      TNL::Containers::StaticVector<Size, Index> coordinates; //, orientation, basis;
};

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
bool HeatmapSolver<Real>::solve(const HeatmapSolver<Real>::Parameters &params) const {
   TNL::Containers::Array<Real, Device> ux(params.xSize * params.ySize),  // data at step u
       aux(params.xSize * params.ySize);                                  // data at step u + 1

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

   Grid<2, int, Real> grid;

   auto init = [=] __cuda_callable__(int i, int j) mutable {
      auto entity = grid.getEntity(i, j);

      auto index = entity.j * xSize + entity.i;

      auto x = entity.i * hx - xDomainSize / 2.;
      auto y = entity.j * hy - yDomainSize / 2.;

      uxView[index] = exp(sigma * (x * x + y * y));
   };

   TNL::Algorithms::ParallelFor2D<Device>::exec(1, 1, params.xSize - 1, params.ySize - 1, init);

   if (!writeGNUPlot("data.txt", params, ux)) return false;

   auto next = [=] __cuda_callable__(int i, int j) mutable {
      auto entity = grid.getEntity(i, j);

      auto index = entity.j * xSize + entity.i;
      auto center = 2 * uxView[index];

      auxView[index] = (uxView[index - 1] - center + uxView[index + 1]) * hx_inv + (uxView[index - xSize] - center + uxView[index + xSize]) * hy_inv;
   };

   auto update = [=] __cuda_callable__(int i) mutable { uxView[i] += auxView[i] * timestep; };

   Real start = 0;

   while (start < params.finalTime) {
      TNL::Algorithms::ParallelFor2D<Device>::exec(1, 1, params.xSize - 1, params.ySize - 1, next);
      TNL::Algorithms::ParallelFor<Device>::exec(0, params.xSize * params.ySize, update);

      start += timestep;
   }

   return writeGNUPlot("data_final.txt", params, ux);
}
