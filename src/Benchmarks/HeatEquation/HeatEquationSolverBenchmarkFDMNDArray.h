// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include <TNL/Containers/NDArray.h>
#include "HeatEquationSolverBenchmark.h"

template< int Dimension,
          typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkFDMNDArray;

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmarkFDMNDArray< 1, Real, Device, Index >: public HeatEquationSolverBenchmark< 1, Real, Device, Index >
{
   static constexpr int Dimension = 1;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using NDArrayType = TNL::Containers::NDArray< Real, TNL::Containers::SizesHolder< Index, 0 >, std::index_sequence< 0 >, Device >;

   TNL::String scheme() { return "fdm"; }

   void init( const Index xSize )
   {
      ux.setSizes( xSize );
      aux.setSizes( xSize );

      ux.setValue( 0 );
      aux.setValue( 0 );

      const Real hx = this->xDomainSize / (Real) xSize;

      auto uxView = ux.getView();
      auto xDomainSize_ = this->xDomainSize;
      auto delta_ = this->delta;
      auto alpha_ = this->alpha;
      auto init = [=] __cuda_callable__( Index i ) mutable
      {
         auto x = i * hx - xDomainSize_ / 2.0;
         uxView( i ) = delta_ * ( 1.0 - TNL::sign( x*x / alpha_ - 1.0 ) );
      };
      TNL::Algorithms::ParallelFor<Device>::exec( 1, xSize - 1, init );
   }

   bool writeGnuplot( const std::string &filename, const Index xSize ) const
   {
      std::ofstream out(filename, std::ios::out);
      if( !out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      for( Index i = 0; i < xSize; i++)
         out << i * hx - this->xDomainSize / 2. << " "
            << ux( i ) << std::endl;
      return out.good();
   }

   void exec( const Index xSize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hx_inv = 1.0 / (hx * hx);

      TNL_ASSERT_EQ( hx, this->xDomainSize / (Real) xSize, "computed wrong hx on the grid" );

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * hx * hx;
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
     {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [=] __cuda_callable__( Index i ) mutable {
            const Real& center = ( Real ) 2.0 * uxView( i );
            auxView( i ) = center + ( uxView( i-1 ) - center + uxView( i + 1 ) ) * hx_inv * timestep;
         };

         //TNL::Algorithms::ParallelFor< Device >::exec( 1, xSize - 1, next );
         ux.forInterior( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:

   NDArrayType ux, aux;
};

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmarkFDMNDArray< 2, Real, Device, Index >: public HeatEquationSolverBenchmark< 2, Real, Device, Index >
{
   static constexpr int Dimension = 2;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using NDArrayType = TNL::Containers::NDArray< Real, TNL::Containers::SizesHolder< Index, 0, 0 >, std::index_sequence< 0, 1 >, Device >;

   TNL::String scheme() { return "fdm"; }

   void init( const Index xSize, const Index ySize )
   {
      ux.setSizes( xSize, ySize );
      aux.setSizes( xSize, ySize );

      ux.setValue( 0 );
      aux.setValue( 0 );

      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;

      auto uxView = ux.getView();
      auto xDomainSize_ = this->xDomainSize;
      auto yDomainSize_ = this->yDomainSize;
      auto delta_ = this->delta;
      auto alpha_ = this->alpha;
      auto beta_ = this->beta;
      auto init = [=] __cuda_callable__( Index i, Index j) mutable
      {
         auto x = i * hx - xDomainSize_ / 2.0;
         auto y = j * hy - yDomainSize_ / 2.0;
         uxView( i, j ) = delta_ * ( 1.0 - TNL::sign( x*x / alpha_ + y*y / beta_ - 1.0 ) );
      };
      TNL::Algorithms::ParallelFor2D<Device>::exec( 1, 1, xSize - 1, ySize - 1, init );
   }

   bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize ) const
   {
      std::ofstream out(filename, std::ios::out);
      if( !out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      for( Index j = 0; j < ySize; j++)
         for( Index i = 0; i < xSize; i++)
            out << i * hx - this->xDomainSize / 2. << " "
               << j * hy - this->yDomainSize / 2. << " "
               << ux( i, j ) << std::endl;
      return out.good();
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
         auto next = [=] __cuda_callable__( Index i, Index j ) mutable
         {
            auto element = uxView( i, j );
            auto center = ( Real ) 2.0 * element;

            auxView( i, j ) = element + ( ( uxView( i-1, j ) - center + uxView( i+1, j ) ) * hx_inv +
                                          ( uxView( i, j-1 ) - center + uxView( i, j+1 ) ) * hy_inv   ) * timestep;
         };

         //TNL::Algorithms::ParallelFor2D< Device >::exec( 1, 1, xSize - 1, ySize - 1, next );
         ux.forInterior( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:

   NDArrayType ux, aux;
};

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmarkFDMNDArray< 3, Real, Device, Index >: public HeatEquationSolverBenchmark< 3, Real, Device, Index >
{
   static constexpr int Dimension = 3;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using NDArrayType = TNL::Containers::NDArray< Real, TNL::Containers::SizesHolder< Index, 0, 0, 0 >, std::index_sequence< 0, 1, 2 >, Device >;

   TNL::String scheme() { return "fdm"; }

   void init( const Index xSize, const Index ySize, const Index zSize )
   {
      ux.setSizes( xSize, ySize, zSize );
      aux.setSizes( xSize, ySize, zSize );

      ux.setValue( 0 );
      aux.setValue( 0 );

      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hz = this->zDomainSize / (Real) zSize;

      auto uxView = ux.getView();
      auto xDomainSize_ = this->xDomainSize;
      auto yDomainSize_ = this->yDomainSize;
      auto zDomainSize_ = this->zDomainSize;
      auto delta_ = this->delta;
      auto alpha_ = this->alpha;
      auto beta_ = this->beta;
      auto gamma_ = this->gamma;
      auto init = [=] __cuda_callable__( Index i, Index j, Index k) mutable
      {
         auto x = i * hx - xDomainSize_ / 2.0;
         auto y = j * hy - yDomainSize_ / 2.0;
         auto z = k * hz - zDomainSize_ / 2.0;
         uxView( i, j, k ) = delta_ * ( 1.0 - TNL::sign( x*x / alpha_ + y*y / beta_ + z*z / gamma_ - 1.0 ) );
      };
      TNL::Algorithms::ParallelFor3D<Device>::exec( 1, 1, 1, xSize - 1, ySize - 1, zSize - 1, init );
   }

   bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize, const Index zSize, const Index zSlice ) const
   {
      std::ofstream out(filename, std::ios::out);
      if( !out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      for( Index j = 0; j < ySize; j++)
         for( Index i = 0; i < xSize; i++)
            out << i * hx - this->xDomainSize / 2. << " "
               << j * hy - this->yDomainSize / 2. << " "
               << ux( i, j, zSlice ) << std::endl;
      return out.good();
   }

   void exec( const Index xSize, const Index ySize, const Index zSize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hz = this->zDomainSize / (Real) zSize;
      const Real hx_inv = 1.0 / (hx * hx);
      const Real hy_inv = 1.0 / (hy * hy);
      const Real hz_inv = 1.0 / (hz * hz);

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx*hx, std::min( hy*hy, hz*hz ) );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [=] __cuda_callable__( Index i, Index j, Index k ) mutable
         {
            auto element = uxView( i, j, k );
            auto center = ( Real ) 2.0 * element;

            auxView( i, j, k ) = element + ( ( uxView( i-1, j, k ) - center + uxView( i+1, j, k ) ) * hx_inv +
                                             ( uxView( i, j-1, k ) - center + uxView( i, j+1, k ) ) * hy_inv +
                                             ( uxView( i, j, k-1 ) - center + uxView( i, j, k+1 ) ) * hz_inv
                                           ) * timestep;
         };
         //TNL::Algorithms::ParallelFor3D< Device >::exec( 1, 1, 1, xSize - 1, ySize - 1, zSize - 1, next );
         ux.forInterior( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }

   }

protected:

   NDArrayType ux, aux;
};
