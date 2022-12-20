// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include "HeatEquationSolverBenchmark.h"
#include <TNL/Meshes/Grid.h>
#include <TNL/Algorithms/ParallelForND.h>

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkParallelForTest : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void exec( const Index xSize, const Index ySize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hx_inv = 1.0 / (hx * hx);
      const Real hy_inv = 1.0 / (hy * hy);

      using GridType = TNL::Meshes::Grid< 2, Real, Device, Index >;
      using Coordinates = typename GridType::CoordinatesType;
      GridType grid;
      grid.setDomain( {0.0,0.0 }, {1.0,1.0} );
      grid.setDimensions( xSize, ySize );
      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min(hx * hx, hy * hy);
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         using GridEntityType = TNL::Meshes::GridEntity< GridType, 2 >;
         auto next = [=] __cuda_callable__( GridEntityType& entity ) mutable
         {
            entity.setGrid( grid );
            //TNL::Meshes::GridEntity< GridType, 2 > entity( grid, c );
            entity.refresh();
            const int& index = entity.getIndex();
            //Coordinates c2 = c;
            //const int& i = c.x();
            //const int& j = c.y();
            //auto index = j * xSize + i;
            auto element = uxView[index];
            auto center = 2 * element;

            auxView[index] = element + ( (uxView[index - 1] -     center + uxView[index + 1]    ) * hx_inv +
                                         (uxView[index - xSize] - center + uxView[index + xSize]) * hy_inv   ) * timestep;
         };

         TNL::Algorithms::ParallelForND< Device, false >::exec( GridEntityType{ 1, 1 }, GridEntityType{ xSize - 1, ySize - 1 } , next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }
};
