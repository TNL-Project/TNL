// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Containers/StaticVector.h>
#include "HeatEquationSolverBenchmark.h"

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
      Grid() = default;

      Grid( const TNL::Containers::StaticVector<Size, Index>& dim ) :
      dimensions( dim ){};

      __cuda_callable__ inline
      Entity<Size, Index, Real> getEntity(Index i, Index j) const {
         Entity<Size, Index, Real> entity(*this);

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
      Entity(const Grid<Size, Index, Real>& grid): grid(grid) {};

      const Grid<Size, Index, Real>& grid;

      Index i, j;
      Index index;
      Index orientation;
      TNL::Containers::StaticVector<Size, Index> coordinates, basis;
};

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkSimpleGrid : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void exec( const Index xSize, const Index ySize ) const
   {
      TNL::Containers::Array<Real, Device> ux( xSize * ySize ),    // data at step u
                                           aux( xSize * ySize );   // data at step u + 1

      // Invalidate ux/aux
      ux = 0;
      aux = 0;

      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hx_inv = 1.0 / (hx * hx);
      const Real hy_inv = 1.0 / (hy * hy);

      auto timestep = this->timeStep ? this->timeStep : std::min(hx * hx, hy * hy);

      auto uxView = ux.getView(), auxView = aux.getView();

      auto xDomainSize_ = this->xDomainSize;
      auto yDomainSize_ = this->yDomainSize;

      auto alpha_ = this->alpha;
      auto beta_ = this->beta;
      auto gamma_ = this->gamma;

      Grid<2, int, Real> grid( {xSize, ySize} );

      auto init = [=] __cuda_callable__(int i, int j) mutable {
         auto entity = grid.getEntity(i, j);

         auto index = entity.j * xSize + entity.i;

         auto x = entity.i * hx - xDomainSize_ / 2.0;
         auto y = entity.j * hy - yDomainSize_ / 2.0;

         uxView[index] = TNL::max( ( x*x / alpha_ )  + ( y*y / beta_ ) + gamma_, 0 );
      };

      TNL::Algorithms::ParallelFor2D<Device>::exec( 1, 1, xSize - 1, ySize - 1, init );

      //if (!writeGNUPlot("data.txt", params, ux)) return false;

      auto next = [=] __cuda_callable__(int i, int j) mutable {
         auto entity = grid.getEntity(i, j);

         auto index = entity.j * xSize + entity.i;
         auto element = uxView[index];
         auto center = 2 * element;

         auxView[index] =  element + ((uxView[index - 1] - center + uxView[index + 1]) * hx_inv +
                                       (uxView[index - xSize] - center + uxView[index + xSize]) * hy_inv) * timestep;
      };

      Real start = 0;

      while (start < this->finalTime) {
         TNL::Algorithms::ParallelFor2D<Device>::exec( 1, 1, xSize - 1, ySize - 1, next);

         uxView = aux.getView();
         auxView = ux.getView();

         start += timestep;
      }

      //return writeGNUPlot("data_final.txt", params, ux);

   };
};
