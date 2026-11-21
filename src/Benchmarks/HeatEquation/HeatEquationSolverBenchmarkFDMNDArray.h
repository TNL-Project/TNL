// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/NDArray.h>
#include <TNL/Algorithms/parallelFor.h>
#include "HeatEquationSolverBenchmark.h"

template< int Dimension, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmarkFDMNDArray;

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkFDMNDArray< 1, Real, Device, Index >
: public HeatEquationSolverBenchmark< 1, Real, Device, Index >
{
   static constexpr int Dimension = 1;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using NDArrayType =
      TNL::Containers::NDArray< Real, TNL::Containers::SizesHolder< Index, 0 >, std::index_sequence< 0 >, Device >;
   using HostNDArrayType =
      TNL::Containers::NDArray< Real, TNL::Containers::SizesHolder< Index, 0 >, std::index_sequence< 0 >, TNL::Devices::Host >;

   TNL::String
   scheme()
   {
      return "fdm";
   }

   void
   init( const Index xSize )
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
      auto init = [ = ] __cuda_callable__( Index i ) mutable
      {
         auto x = i * hx - xDomainSize_ / 2.0;
         uxView( i ) = HeatEquationSolverBenchmarkBase< Real, Index >::getInitialCondition( x, 0, 0, alpha_, 1, 1, delta_ );
      };
      TNL::Algorithms::parallelFor< Device >( 1, xSize - 1, init );
   }

   void
   getSolution( VectorType& u ) const
   {
      // the solution is copied to the host, because it may be stored on the GPU
      HostNDArrayType hostUx;
      hostUx = ux;
      const Index xSize = hostUx.template getSize< 0 >();
      TNL::Containers::Vector< Real, TNL::Devices::Host, Index > hostU( xSize );
      for( Index i = 0; i < xSize; i++ )
         hostU[ i ] = hostUx( i );
      u = hostU;
   }

   void
   exec( const Index xSize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hx_inv = 1.0 / ( hx * hx );

      TNL_ASSERT_EQ( hx, this->xDomainSize / (Real) xSize, "computed wrong hx on the grid" );

      Real start = 0;
      Index iterations = 0;
      // the host does not wait for each kernel; Benchmark::time synchronizes the device after the time loop
      const auto launchConfiguration = TNL::Benchmarks::getAsyncLaunchConfiguration< Device >();
      const Real timestep = this->timeStep ? this->timeStep : 0.1 * hx * hx;
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [ = ] __cuda_callable__( Index i ) mutable
         {
            auto element = uxView( i );
            auto center = (Real) 2.0 * element;
            auxView( i ) = element + ( ( uxView( i - 1 ) - center + uxView( i + 1 ) ) * hx_inv ) * timestep;
         };

         //TNL::Algorithms::ParallelFor< Device >::exec( 1, xSize - 1, next );
         ux.forInterior( next, launchConfiguration );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   NDArrayType ux, aux;
};

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkFDMNDArray< 2, Real, Device, Index >
: public HeatEquationSolverBenchmark< 2, Real, Device, Index >
{
   static constexpr int Dimension = 2;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using NDArrayType =
      TNL::Containers::NDArray< Real, TNL::Containers::SizesHolder< Index, 0, 0 >, std::index_sequence< 0, 1 >, Device >;
   using HostNDArrayType = TNL::Containers::
      NDArray< Real, TNL::Containers::SizesHolder< Index, 0, 0 >, std::index_sequence< 0, 1 >, TNL::Devices::Host >;
   using CoordinatesType = TNL::Containers::StaticArray< Dimension, Index >;

   TNL::String
   scheme()
   {
      return "fdm";
   }

   void
   init( const Index xSize, const Index ySize )
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
      auto init = [ = ] __cuda_callable__( const CoordinatesType& idx ) mutable
      {
         const Index& i = idx.x();
         const Index& j = idx.y();
         auto x = i * hx - xDomainSize_ / 2.0;
         auto y = j * hy - yDomainSize_ / 2.0;
         uxView( i, j ) =
            HeatEquationSolverBenchmarkBase< Real, Index >::getInitialCondition( x, y, 0, alpha_, beta_, 1, delta_ );
      };
      TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1 }, init );
   }

   void
   getSolution( VectorType& u ) const
   {
      // the solution is copied to the host, because it may be stored on the GPU, and reordered,
      // because the last index of the array changes fastest
      HostNDArrayType hostUx;
      hostUx = ux;
      const Index xSize = hostUx.template getSize< 0 >();
      const Index ySize = hostUx.template getSize< 1 >();
      TNL::Containers::Vector< Real, TNL::Devices::Host, Index > hostU( xSize * ySize );
      for( Index j = 0; j < ySize; j++ )
         for( Index i = 0; i < xSize; i++ )
            hostU[ j * xSize + i ] = hostUx( i, j );
      u = hostU;
   }

   void
   exec( const Index xSize, const Index ySize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hx_inv = 1.0 / ( hx * hx );
      const Real hy_inv = 1.0 / ( hy * hy );

      Real start = 0;
      Index iterations = 0;
      // the host does not wait for each kernel; Benchmark::time synchronizes the device after the time loop
      const auto launchConfiguration = TNL::Benchmarks::getAsyncLaunchConfiguration< Device >();
      const Real timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx * hx, hy * hy );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [ = ] __cuda_callable__( Index i, Index j ) mutable
         {
            //const Index& i = idx.x();
            //const Index& j = idx.y();
            auto element = uxView( i, j );
            auto center = (Real) 2.0 * element;

            auxView( i, j ) = element
                            + ( ( uxView( i - 1, j ) - center + uxView( i + 1, j ) ) * hx_inv
                                + ( uxView( i, j - 1 ) - center + uxView( i, j + 1 ) ) * hy_inv )
                                 * timestep;
         };

         //TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1 }, next );
         ux.forInterior( next, launchConfiguration );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   NDArrayType ux, aux;
};

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkFDMNDArray< 3, Real, Device, Index >
: public HeatEquationSolverBenchmark< 3, Real, Device, Index >
{
   static constexpr int Dimension = 3;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using NDArrayType =
      TNL::Containers::NDArray< Real, TNL::Containers::SizesHolder< Index, 0, 0, 0 >, std::index_sequence< 0, 1, 2 >, Device >;
   using HostNDArrayType = TNL::Containers::
      NDArray< Real, TNL::Containers::SizesHolder< Index, 0, 0, 0 >, std::index_sequence< 0, 1, 2 >, TNL::Devices::Host >;
   using CoordinatesType = TNL::Containers::StaticArray< Dimension, Index >;

   TNL::String
   scheme()
   {
      return "fdm";
   }

   void
   init( const Index xSize, const Index ySize, const Index zSize )
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
      auto init = [ = ] __cuda_callable__( const CoordinatesType& idx ) mutable
      {
         const Index& i = idx.x();
         const Index& j = idx.y();
         const Index& k = idx.z();
         auto x = i * hx - xDomainSize_ / 2.0;
         auto y = j * hy - yDomainSize_ / 2.0;
         auto z = k * hz - zDomainSize_ / 2.0;
         uxView( i, j, k ) =
            HeatEquationSolverBenchmarkBase< Real, Index >::getInitialCondition( x, y, z, alpha_, beta_, gamma_, delta_ );
      };
      TNL::Algorithms::parallelFor< Device >(
         CoordinatesType{ 1, 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1, zSize - 1 }, init );
   }

   void
   getSolution( VectorType& u ) const
   {
      // the solution is copied to the host, because it may be stored on the GPU, and reordered,
      // because the last index of the array changes fastest
      HostNDArrayType hostUx;
      hostUx = ux;
      const Index xSize = hostUx.template getSize< 0 >();
      const Index ySize = hostUx.template getSize< 1 >();
      const Index zSize = hostUx.template getSize< 2 >();
      TNL::Containers::Vector< Real, TNL::Devices::Host, Index > hostU( xSize * ySize * zSize );
      for( Index k = 0; k < zSize; k++ )
         for( Index j = 0; j < ySize; j++ )
            for( Index i = 0; i < xSize; i++ )
               hostU[ ( k * ySize + j ) * xSize + i ] = hostUx( i, j, k );
      u = hostU;
   }

   void
   exec( const Index xSize, const Index ySize, const Index zSize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hz = this->zDomainSize / (Real) zSize;
      const Real hx_inv = 1.0 / ( hx * hx );
      const Real hy_inv = 1.0 / ( hy * hy );
      const Real hz_inv = 1.0 / ( hz * hz );

      Real start = 0;
      Index iterations = 0;
      // the host does not wait for each kernel; Benchmark::time synchronizes the device after the time loop
      const auto launchConfiguration = TNL::Benchmarks::getAsyncLaunchConfiguration< Device >();
      const Real timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx * hx, std::min( hy * hy, hz * hz ) );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [ = ] __cuda_callable__( Index i, Index j, Index k ) mutable
         {
            //const Index& i = idx.x();
            //const Index& j = idx.y();
            //const Index& k = idx.z();
            auto element = uxView( i, j, k );
            auto center = (Real) 2.0 * element;

            auxView( i, j, k ) = element
                               + ( ( uxView( i - 1, j, k ) - center + uxView( i + 1, j, k ) ) * hx_inv
                                   + ( uxView( i, j - 1, k ) - center + uxView( i, j + 1, k ) ) * hy_inv
                                   + ( uxView( i, j, k - 1 ) - center + uxView( i, j, k + 1 ) ) * hz_inv )
                                    * timestep;
         };
         //TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1, zSize - 1
         //}, next );
         ux.forInterior( next, launchConfiguration );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   NDArrayType ux, aux;
};
