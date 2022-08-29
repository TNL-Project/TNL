// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Meshes/Grid.h>
#include "HeatEquationSolverBenchmark.h"

template< int Size = 1,
          typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkGridShmem : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void exec( const Index xSize, const Index ySize )
   {
      using Grid2D = TNL::Meshes::Grid<2, Real, Device, int>;

      Grid2D grid;

      // Grid implementation defines its dimensions in the amount of edges.
      // To align it size to all other benchmarks substract 1
      grid.setDimensions( xSize - 1, ySize - 1 );
      grid.setDomain( { 0.0, 0.0}, { this->xDomainSize, this->yDomainSize } );

      const Real hx_inv = grid.template getSpaceStepsProducts<-2, 0>();
      const Real hy_inv = grid.template getSpaceStepsProducts<0, -2>();

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ?
         this->timeStep : std::min( grid.template getSpaceStepsProducts<2, 0>(), grid.template getSpaceStepsProducts<0, 2>() );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
     {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto width = grid.getDimensions().x() + 1;
         auto next = [=] __cuda_callable__(const typename Grid2D::template EntityType<0>&entity) mutable {
            auto index = entity.getIndex();
            auto element = uxView[index];
            auto center = 2 * element;

            TNL::Containers::StaticVector< Size, Real > v( 1.0 );
            auxView[index] = element + ( ( uxView[index - 1] - center + uxView[index + 1] ) * hx_inv +
                                         ( uxView[index - width] - center + uxView[index + width] ) * hy_inv ) *
                                         TNL::l1Norm( v ) * timestep;
         };

         grid.template forInteriorEntities<0>( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   };
};
