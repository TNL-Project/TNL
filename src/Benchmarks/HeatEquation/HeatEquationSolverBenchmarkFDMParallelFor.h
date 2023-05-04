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

template< int Dimension,
          typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkFDMParallelFor;

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmarkFDMParallelFor< 1, Real, Device, Index > : public HeatEquationSolverBenchmark< 1, Real, Device, Index >
{
   static constexpr int Dimension = 1;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;

   TNL::String scheme() { return "fdm"; }

   void init( const Index xSize )
   {
      BaseBenchmarkType::init( xSize, ux, aux );
   }

   bool writeGnuplot( const std::string &filename, const Index xSize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

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

         TNL::Algorithms::parallelFor< Device >( 1, xSize - 1, next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:

   VectorType ux, aux;
};

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmarkFDMParallelFor< 2, Real, Device, Index > : public HeatEquationSolverBenchmark< 2, Real, Device, Index >
{
   static constexpr int Dimension = 2;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using CoordinatesType = TNL::Containers::StaticArray< Dimension, Index >;

   TNL::String scheme() { return "fdm"; }

   void init( const Index xSize, const Index ySize )
   {
      BaseBenchmarkType::init( xSize, ySize, ux, aux );
   }

   bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize );
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
         auto next = [=] __cuda_callable__( const CoordinatesType& i ) mutable
         {
            auto index = i.y() * xSize + i.x();
            auto element = uxView[index];
            auto center = ( Real ) 2.0 * element;

            auxView[index] = element + ( (uxView[index - 1] -     center + uxView[index + 1]    ) * hx_inv +
                                         (uxView[index - xSize] - center + uxView[index + xSize]) * hy_inv   ) * timestep;
         };
         TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1 }, next );

         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:

   VectorType ux, aux;
};

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmarkFDMParallelFor< 3, Real, Device, Index > : public HeatEquationSolverBenchmark< 3, Real, Device, Index >
{
   static constexpr int Dimension = 3;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using CoordinatesType = TNL::Containers::StaticArray< Dimension, Index >;

   TNL::String scheme() { return "fdm"; }

   void init( const Index xSize, const Index ySize, const Index zSize )
   {
      BaseBenchmarkType::init( xSize, ySize, zSize, ux, aux );
   }

   bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize, const Index zSize, const Index zSlice ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize, zSize, zSlice );
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
         auto next = [=] __cuda_callable__( const CoordinatesType& idx ) mutable
         {
            auto index = ( idx.z() * ySize + idx.y() ) * xSize + idx.x();
            auto element = uxView[index];
            auto center = ( Real ) 2.0 * element;

            auxView[index] = element + ( ( uxView[ index-1 ] -      center + uxView[ index+1 ]      ) * hx_inv +
                                         ( uxView[ index-xSize ] -  center + uxView[ index+xSize ]  ) * hy_inv +
                                         ( uxView[ index-xySize ] - center + uxView[ index+xySize ] ) * hz_inv
                                       ) * timestep;
         };
         TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1, zSize - 1 }, next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:

   VectorType ux, aux;
};
