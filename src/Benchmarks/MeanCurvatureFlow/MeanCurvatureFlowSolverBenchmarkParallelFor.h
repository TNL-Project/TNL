// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Benchmarks/Devices.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>

#include "MeanCurvatureFlowSolverBenchmark.h"

/**
 * \brief Mean-curvature flow solved with \ref TNL::Algorithms::parallelFor over multi-indices.
 *
 * The finite difference scheme and the explicit Euler method are the same as in \ref MeanCurvatureFlowFDMGrid, but the
 * cells are traversed by `parallelFor` and their indices are computed directly from the multi-index and the strides
 * of the row-major layout, without the grid.
 */
template< int Dimension, typename Real, typename Device, typename Index >
class MeanCurvatureFlowParallelForSolver
{
public:
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using Sizes = TNL::Containers::StaticVector< Dimension, Index >;
   using Steps = TNL::Containers::StaticVector< Dimension, Real >;
   // parallelFor passes a plain index in 1D and a multi-index otherwise
   using MultiIndex = std::conditional_t< Dimension == 1, Index, Sizes >;

   void
   init( const Sizes& sizes, const VectorType& u )
   {
      this->sizes = sizes;
      strides[ 0 ] = 1;
      for( int d = 1; d < Dimension; d++ )
         strides[ d ] = strides[ d - 1 ] * sizes[ d - 1 ];
      // the auxiliary vector must have the same boundary values as the solution
      uNew = u;
      q.setSize( u.getSize() );
      qBar.setSize( u.getSize() );
      // the values on the boundary are not computed
      q = 1.0;
      qBar = 1.0;
   }

   void
   solve( VectorType& u, const Steps& domainSizes, Real timeStep, Real finalTime, Index maxIterations )
   {
      Steps h;
      Real hMin = domainSizes[ 0 ] / sizes[ 0 ];
      for( int d = 0; d < Dimension; d++ ) {
         h[ d ] = domainSizes[ d ] / sizes[ d ];
         hMin = TNL::min( hMin, h[ d ] );
      }
      const Real tau = timeStep > 0 ? timeStep : 0.1 * hMin * hMin;

      Real time = 0;
      Index iterations = 0;
      while( time < finalTime && ( maxIterations == 0 || iterations < maxIterations ) ) {
         computeGradientNorms( u, h );
         explicitStep( u, h, tau );
         u.swap( uNew );
         time += tau;
         iterations++;
      }
   }

   /**
    * \brief Returns the distance of the neighbor cells along the given axis.
    *
    * The stride along the x axis is the constant one, so that the compiler can vectorize the loops along the x axis
    * (GCC 13 does not vectorize them when the stride is loaded from the array, which made the solver 1.6 times slower).
    */
   __cuda_callable__
   static Index
   getStride( const Sizes& strides, int axis )
   {
      return axis == 0 ? 1 : strides[ axis ];
   }

   __cuda_callable__
   static Index
   getLinearIndex( const MultiIndex& c, const Sizes& strides )
   {
      if constexpr( Dimension == 1 )
         return c;
      else {
         Index result = c[ 0 ];
         for( int d = 1; d < Dimension; d++ )
            result += c[ d ] * strides[ d ];
         return result;
      }
   }

   // the methods with extended lambdas must be public for nvcc
   void
   computeGradientNorms( const VectorType& u, const Steps& h )
   {
      const Steps hInv = 1.0 / h;
      const Sizes strides = this->strides;
      const auto uView = u.getConstView();
      auto qView = q.getView();
      auto qBarView = qBar.getView();
      auto kernel = [ = ] __cuda_callable__( const MultiIndex& idx ) mutable
      {
         const Index c = getLinearIndex( idx, strides );
         const Real u_c = uView[ c ];
         Real forwardSum = 0;
         Real backwardSum = 0;
         for( int i = 0; i < Dimension; i++ ) {
            const Real forward = ( uView[ c + getStride( strides, i ) ] - u_c ) * hInv[ i ];
            const Real backward = ( u_c - uView[ c - getStride( strides, i ) ] ) * hInv[ i ];
            forwardSum += forward * forward;
            backwardSum += backward * backward;
         }
         qView[ c ] = TNL::sqrt( Real( 1 ) + forwardSum );
         qBarView[ c ] = TNL::sqrt( Real( 1 ) + Real( 0.5 ) * ( forwardSum + backwardSum ) );
      };
      // the host does not wait for the kernel; Benchmark::time synchronizes the device after the time loop
      TNL::Algorithms::parallelFor< Device >(
         Sizes( 1 ), Sizes( sizes - 1 ), TNL::Benchmarks::getAsyncLaunchConfiguration< Device >(), kernel );
   }

   void
   explicitStep( const VectorType& u, const Steps& h, Real tau )
   {
      const Steps hSqrInv = 1.0 / ( h * h );
      const Sizes strides = this->strides;
      const auto uView = u.getConstView();
      const auto qView = q.getConstView();
      const auto qBarView = qBar.getConstView();
      auto uNewView = uNew.getView();
      auto kernel = [ = ] __cuda_callable__( const MultiIndex& idx ) mutable
      {
         const Index c = getLinearIndex( idx, strides );
         const Real u_c = uView[ c ];
         Real divergence = 0;
         for( int i = 0; i < Dimension; i++ ) {
            const Index east = c + getStride( strides, i );
            const Index west = c - getStride( strides, i );
            divergence += ( ( uView[ east ] - u_c ) / qView[ c ] - ( u_c - uView[ west ] ) / qView[ west ] ) * hSqrInv[ i ];
         }
         uNewView[ c ] = u_c + tau * qBarView[ c ] * divergence;
      };
      // the host does not wait for the kernel; Benchmark::time synchronizes the device after the time loop
      TNL::Algorithms::parallelFor< Device >(
         Sizes( 1 ), Sizes( sizes - 1 ), TNL::Benchmarks::getAsyncLaunchConfiguration< Device >(), kernel );
   }

protected:
   Sizes sizes, strides;
   VectorType uNew, q, qBar;
};

/**
 * \brief Mean-curvature flow benchmark with the explicit finite difference scheme implemented by `parallelFor`.
 */
template< int Dimension, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct MeanCurvatureFlowSolverBenchmarkParallelFor;

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkParallelFor< 1, Real, Device, Index >
: public MeanCurvatureFlowSolverBenchmark< 1, Real, Device, Index >
{
   using BaseBenchmarkType = MeanCurvatureFlowSolverBenchmark< 1, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;

   TNL::String
   scheme() override
   {
      return "fdm";
   }

   void
   init( const Index xSize ) override
   {
      BaseBenchmarkType::init( xSize, ux, aux );
      solver.init( { xSize }, ux );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize ) const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

   void
   exec( const Index xSize ) override
   {
      solver.solve( ux, { this->xDomainSize }, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   VectorType ux, aux;
   MeanCurvatureFlowParallelForSolver< 1, Real, Device, Index > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkParallelFor< 2, Real, Device, Index >
: public MeanCurvatureFlowSolverBenchmark< 2, Real, Device, Index >
{
   using BaseBenchmarkType = MeanCurvatureFlowSolverBenchmark< 2, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;

   TNL::String
   scheme() override
   {
      return "fdm";
   }

   void
   init( const Index xSize, const Index ySize ) override
   {
      BaseBenchmarkType::init( xSize, ySize, ux, aux );
      solver.init( { xSize, ySize }, ux );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize ) const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize );
   }

   void
   exec( const Index xSize, const Index ySize ) override
   {
      solver.solve( ux, { this->xDomainSize, this->yDomainSize }, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   VectorType ux, aux;
   MeanCurvatureFlowParallelForSolver< 2, Real, Device, Index > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkParallelFor< 3, Real, Device, Index >
: public MeanCurvatureFlowSolverBenchmark< 3, Real, Device, Index >
{
   using BaseBenchmarkType = MeanCurvatureFlowSolverBenchmark< 3, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;

   TNL::String
   scheme() override
   {
      return "fdm";
   }

   void
   init( const Index xSize, const Index ySize, const Index zSize ) override
   {
      BaseBenchmarkType::init( xSize, ySize, zSize, ux, aux );
      solver.init( { xSize, ySize, zSize }, ux );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize, const Index zSize, const Index zSlice )
      const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize, zSize, zSlice );
   }

   void
   exec( const Index xSize, const Index ySize, const Index zSize ) override
   {
      solver.solve(
         ux,
         { this->xDomainSize, this->yDomainSize, this->zDomainSize },
         this->timeStep,
         this->finalTime,
         this->maxIterations );
   }

protected:
   VectorType ux, aux;
   MeanCurvatureFlowParallelForSolver< 3, Real, Device, Index > solver;
};
