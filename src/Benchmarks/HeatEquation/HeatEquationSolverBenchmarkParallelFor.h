// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>

#include "HeatEquationSolverBenchmark.h"

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkParallelFor : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void exec( const Index xSize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hx_inv = 1.0 / (hx * hx);

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * hx*hx;
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [=] __cuda_callable__( Index i ) mutable
         {
            auto element = uxView[i];
            auto center = ( Real ) 2.0 * element;

            auxView[ i ] = element + ( (uxView[ i-1 ] - center + uxView[ i+1 ] ) * hx_inv ) * timestep;
         };

         TNL::Algorithms::ParallelFor< Device >::exec( 1, xSize - 1, next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

   void exec( const Index xSize, const Index ySize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hx_inv = 1.0 / (hx * hx);
      const Real hy_inv = 1.0 / (hy * hy);

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min(hx * hx, hy * hy);
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [=] __cuda_callable__( const TNL::Containers::StaticArray< 2, int >& i ) mutable
         {
            auto index = i.y() * xSize + i.x();
            auto element = uxView[index];
            auto center = ( Real ) 2.0 * element;

            auxView[index] = element + ( (uxView[index - 1] -     center + uxView[index + 1]    ) * hx_inv +
                                         (uxView[index - xSize] - center + uxView[index + xSize]) * hy_inv   ) * timestep;
         };

         const TNL::Containers::StaticArray< 2, int > begin = { 1, 1 };
         const TNL::Containers::StaticArray< 2, int > end = { xSize - 1, ySize - 1 };
         TNL::Algorithms::parallelFor< Device >( begin, end, next );

         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

   void exec( const Index xSize, const Index ySize, const Index zSize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hz = this->zDomainSize / (Real) zSize;
      const Real hx_inv = 1.0 / (hx * hx);
      const Real hy_inv = 1.0 / (hy * hy);
      const Real hz_inv = 1.0 / (hz * hz);
      const auto xySize = xSize * ySize;

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx*hx, std::min( hy*hy, hz*hz ) );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [=] __cuda_callable__( Index i, Index j, Index k ) mutable
         {
            auto index = ( k * ySize + j ) * xSize + i;
            auto element = uxView[index];
            auto center = ( Real ) 2.0 * element;

            auxView[index] = element + ( ( uxView[ index-1 ] -      center + uxView[ index+1 ]      ) * hx_inv +
                                         ( uxView[ index-xSize ] -  center + uxView[ index+xSize ]  ) * hy_inv +
                                         ( uxView[ index-xySize ] - center + uxView[ index+xySize ] ) * hz_inv
                                       ) * timestep;
         };

         TNL::Algorithms::ParallelFor3D< Device >::exec( 1, 1, 1, xSize - 1, ySize - 1, zSize - 1, next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }
};
