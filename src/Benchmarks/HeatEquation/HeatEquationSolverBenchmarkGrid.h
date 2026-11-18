// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>
#include "HeatEquationSolverBenchmark.h"

template< typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmarkGrid : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void
   exec( const Index xSize, const Index ySize ) override
   {
      using Grid2D = TNL::Meshes::Grid< 2, Real, Device, int >;

      Grid2D grid;

      grid.setSizes( { xSize, ySize } );
      grid.setDomain( { 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize } );

      const Real hx = grid.getSpaceSteps().x();
      const Real hy = grid.getSpaceSteps().y();
      const Real hx_inv = 1.0 / ( hx * hx );
      const Real hy_inv = 1.0 / ( hy * hy );

      TNL_ASSERT_EQ( hx, this->xDomainSize / (Real) xSize, "computed wrong hx on the grid" );
      TNL_ASSERT_EQ( hy, this->yDomainSize / (Real) ySize, "computed wrong hy on the grid" );

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx * hx, hy * hy );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto width = grid.getSizes().x();
         auto next = [ = ] __cuda_callable__( const typename Grid2D::template EntityType< 2 >& entity ) mutable
         {
            auto index = entity.getIndex();
            auto element = uxView[ index ];
            auto center = 2 * element;

            auxView[ index ] = element
                             + ( ( uxView[ index - 1 ] - center + uxView[ index + 1 ] ) * hx_inv
                                 + ( uxView[ index - width ] - center + uxView[ index + width ] ) * hy_inv )
                                  * timestep;
         };

         grid.template forInteriorEntities< 2 >( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }
};
