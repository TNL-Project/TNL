// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include "HeatEquationSolverBenchmark.h"

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkParallelFor : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void exec( const Index xSize, const Index ySize ) const
   {
      TNL::Containers::Array<Real, Device> ux( xSize *  ySize ); // data at step u
      TNL::Containers::Array<Real, Device> aux( xSize * ySize );// data at step u + 1

      // Invalidate ux/aux
      ux = 0;
      aux = 0;

      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hx_inv = 1. / (hx * hx);
      const Real hy_inv = 1. / (hy * hy);

      auto timestep = this->timeStep ? this->timeStep : std::min(hx * hx, hy * hy);

      auto uxView = ux.getView(), auxView = aux.getView();

      auto xDomainSize_ = this->xDomainSize;
      auto yDomainSize_ = this->yDomainSize;
      auto alpha_ = this->alpha;
      auto beta_ = this->beta;
      auto gamma_ = this->gamma;
      auto init = [=] __cuda_callable__(int i, int j) mutable
      {
         auto index = j * xSize + i;

         auto x = i * hx - xDomainSize_ / 2.;
         auto y = j * hy - yDomainSize_ / 2.;

         uxView[index] = TNL::max( ( x*x / alpha_ )  + ( y*y / beta_ ) + gamma_, 0.0 );
      };

      TNL::Algorithms::ParallelFor2D<Device>::exec( 1, 1, xSize - 1, ySize - 1, init );

      //if (!writeGNUPlot("data.txt", params, ux))
      //   return false;

      auto next = [=] __cuda_callable__( Index i, Index j ) mutable
      {
         auto index = j * xSize + i;
         auto element = uxView[index];
         auto center = 2 * element;

         auxView[index] = element + ((uxView[index - 1] - center + uxView[index + 1]) * hx_inv +
                                    (uxView[index - xSize] - center + uxView[index + xSize]) * hy_inv) * timestep;
      };

      Real start = 0;

      while( start < this->finalTime )
      {
         TNL::Algorithms::ParallelFor2D< Device >::exec( 1, 1, xSize - 1, ySize - 1, next );

         uxView = aux.getView();
         auxView = ux.getView();

         start += timestep;
      }

      //return writeGNUPlot("data_final.txt", params, ux);

   };
};
