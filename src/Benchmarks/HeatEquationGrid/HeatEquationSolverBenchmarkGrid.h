// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Meshes/Grid.h>
#include "HeatEquationSolverBenchmark.h"

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkGrid : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void exec( const Index xSize, const Index ySize ) const
   {
      using Grid2D = TNL::Meshes::Grid<2, Real, Device, int>;

      Grid2D grid;

      // Grid implementation defines its dimensions in the amount of edges.
      // To align it size to all other benchmarks substract 1
      grid.setDimensions( xSize - 1, ySize - 1 );
      grid.setDomain( { 0.0, 0.0}, { this->xDomainSize, this->yDomainSize } );

      auto verticesCount = grid.template getEntitiesCount<0>();

      TNL::Containers::Array<Real, Device> ux(verticesCount), // data at step u
                                           aux(verticesCount);// data at step u + 1

      // Invalidate ux/aux
      ux = 0;
      aux = 0;

      const Real hx = grid.template getSpaceStepsProducts<1, 0>();
      const Real hy = grid.template getSpaceStepsProducts<0, 1>();
      const Real hx_inv = grid.template getSpaceStepsProducts<-2, 0>();
      const Real hy_inv = grid.template getSpaceStepsProducts<0, -2>();

      auto timestep = this->timeStep ?
         this->timeStep : std::min( grid.template getSpaceStepsProducts<2, 0>(), grid.template getSpaceStepsProducts<0, 2>() );

      auto uxView = ux.getView(),
         auxView = aux.getView();

      auto xDomainSize_ = this->xDomainSize;
      auto yDomainSize_ = this->yDomainSize;

      auto alpha_ = this->alpha;
      auto beta_  = this->beta;
      auto gamma_ = this->gamma;

      auto init = [=] __cuda_callable__(const typename Grid2D::template EntityType<0> &entity) mutable {
         auto index = entity.getIndex();

         auto x = entity.getCoordinates().x() * hx - xDomainSize_ / 2.;
         auto y = entity.getCoordinates().y() * hy - yDomainSize_ / 2.;

         uxView[index] = TNL::max( ( x*x / alpha_ )  + ( y*y / beta_ ) + gamma_, 0 );
      };

      grid.template forInterior<0>(init);

      //if (!writeGNUPlot("data.txt", params, ux))
      //   return false;

      auto width = grid.getDimensions().x() + 1;
      auto next = [=] __cuda_callable__(const typename Grid2D::template EntityType<0>&entity) mutable {
         auto index = entity.getIndex();
         auto element = uxView[index];
         auto center = 2 * element;

         auxView[index] = element + ((uxView[index - 1] - center + uxView[index + 1]) * hx_inv +
                                    (uxView[index - width] - center + uxView[index + width]) * hy_inv) * timestep;
      };

      Real start = 0;

      while (start < this->finalTime) {
         grid.template forInterior<0>(next);

         uxView = aux.getView();
         auxView = ux.getView();

         start += timestep;
      }
      //return writeGNUPlot("data_final.txt", params, ux);
   };
};
