// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "MeanCurvatureFlowSolverBenchmark.h"
#include "../GridLibraries/KokkosSupport.h"

#ifdef TNL_BENCHMARKS_HAVE_KOKKOS

   #include <algorithm>
   #include <array>
   #include <utility>

/**
 * \brief Mean-curvature flow solved with Kokkos for the comparison with TNL.
 *
 * The finite difference scheme and the explicit Euler method are the same as in \ref MeanCurvatureFlowFDMGrid. The
 * solution is stored in the TNL vectors initialized by the base benchmark class and Kokkos accesses them through
 * unmanaged views. The approximations \f$q\f$ and \f$\bar q\f$ of the gradient norm are stored in Kokkos views.
 */
template< int Dimension, typename Real, typename Device, typename Index >
class MeanCurvatureFlowKokkosSolver
{
public:
   using ExecSpace = typename TNL::Benchmarks::KokkosSupport::ExecutionSpace< Device >::type;
   using View = TNL::Benchmarks::KokkosSupport::View< Real, Dimension, ExecSpace >;
   using UnmanagedView = TNL::Benchmarks::KokkosSupport::UnmanagedView< Real, Dimension, ExecSpace >;
   using Sizes = std::array< int, Dimension >;
   using Steps = std::array< Real, Dimension >;

   void
   init( const Sizes& sizes )
   {
      this->sizes = sizes;
      q = TNL::Benchmarks::KokkosSupport::createView< View, Dimension >( "q", sizes );
      qBar = TNL::Benchmarks::KokkosSupport::createView< View, Dimension >( "qBar", sizes );
      // the values on the boundary are not computed
      Kokkos::deep_copy( q, 1.0 );
      Kokkos::deep_copy( qBar, 1.0 );
   }

   template< typename Vector >
   void
   solve( Vector& u, Vector& uNew, const Steps& domainSizes, Real timeStep, Real finalTime, Index maxIterations )
   {
      Steps h;
      Real hMin = domainSizes[ 0 ] / sizes[ 0 ];
      for( int d = 0; d < Dimension; d++ ) {
         h[ d ] = domainSizes[ d ] / sizes[ d ];
         hMin = std::min( hMin, h[ d ] );
      }
      const Real tau = timeStep > 0 ? timeStep : 0.1 * hMin * hMin;

      auto uView = TNL::Benchmarks::KokkosSupport::wrap< UnmanagedView, Dimension >( u.getData(), sizes );
      auto uNewView = TNL::Benchmarks::KokkosSupport::wrap< UnmanagedView, Dimension >( uNew.getData(), sizes );
      Real time = 0;
      Index iterations = 0;
      while( time < finalTime && ( ! maxIterations || iterations < maxIterations ) ) {
         computeGradientNorms( uView, h );
         explicitStep( uView, uNewView, h, tau );
         std::swap( uView, uNewView );
         time += tau;
         iterations++;
      }
      ExecSpace().fence();
      // the TNL vectors must hold the solution in the same vector as in the other implementations
      if( iterations % 2 == 1 )
         u.swap( uNew );
   }

   // the methods with Kokkos lambdas must be public for nvcc
   void
   computeGradientNorms( const UnmanagedView& u, const Steps& h ) const
   {
      using TNL::Benchmarks::KokkosSupport::at;
      using TNL::Benchmarks::KokkosSupport::shifted;
      Kokkos::Array< Real, Dimension > hInv;
      for( int d = 0; d < Dimension; d++ )
         hInv[ d ] = 1.0 / h[ d ];
      const View q = this->q;
      const View qBar = this->qBar;
      const auto [ begin, end ] = TNL::Benchmarks::KokkosSupport::getInterior< Dimension >( sizes );
      TNL::Benchmarks::KokkosSupport::forCells< ExecSpace, Dimension >(
         "mean-curvature-flow-gradient-norms", begin, end, KOKKOS_LAMBDA( const Kokkos::Array< int, Dimension >& c ) {
            const Real u_c = at( u, c );
            Real forwardSum = 0;
            Real backwardSum = 0;
            for( int i = 0; i < Dimension; i++ ) {
               const Real forward = ( at( u, shifted( c, i, 1 ) ) - u_c ) * hInv[ i ];
               const Real backward = ( u_c - at( u, shifted( c, i, -1 ) ) ) * hInv[ i ];
               forwardSum += forward * forward;
               backwardSum += backward * backward;
            }
            at( q, c ) = Kokkos::sqrt( Real( 1 ) + forwardSum );
            at( qBar, c ) = Kokkos::sqrt( Real( 1 ) + Real( 0.5 ) * ( forwardSum + backwardSum ) );
         } );
   }

   void
   explicitStep( const UnmanagedView& u, const UnmanagedView& uNew, const Steps& h, Real tau ) const
   {
      using TNL::Benchmarks::KokkosSupport::at;
      using TNL::Benchmarks::KokkosSupport::shifted;
      Kokkos::Array< Real, Dimension > hSqrInv;
      for( int d = 0; d < Dimension; d++ )
         hSqrInv[ d ] = 1.0 / ( h[ d ] * h[ d ] );
      const View q = this->q;
      const View qBar = this->qBar;
      const auto [ begin, end ] = TNL::Benchmarks::KokkosSupport::getInterior< Dimension >( sizes );
      TNL::Benchmarks::KokkosSupport::forCells< ExecSpace, Dimension >(
         "mean-curvature-flow-explicit-step", begin, end, KOKKOS_LAMBDA( const Kokkos::Array< int, Dimension >& c ) {
            const Real u_c = at( u, c );
            const Real q_c = at( q, c );
            Real divergence = 0;
            for( int i = 0; i < Dimension; i++ ) {
               const auto west = shifted( c, i, -1 );
               divergence +=
                  ( ( at( u, shifted( c, i, 1 ) ) - u_c ) / q_c - ( u_c - at( u, west ) ) / at( q, west ) ) * hSqrInv[ i ];
            }
            at( uNew, c ) = u_c + tau * at( qBar, c ) * divergence;
         } );
   }

protected:
   Sizes sizes;
   View q, qBar;
};

/**
 * \brief Mean-curvature flow benchmark with the Kokkos implementation of the explicit finite difference scheme.
 */
template< int Dimension, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct MeanCurvatureFlowSolverBenchmarkKokkos;

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkKokkos< 1, Real, Device, Index >
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
      // the auxiliary vector must have the same boundary values as the solution
      aux = ux;
      solver.init( { xSize } );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize ) const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

   void
   exec( const Index xSize ) override
   {
      solver.solve( ux, aux, { this->xDomainSize }, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   VectorType ux, aux;
   MeanCurvatureFlowKokkosSolver< 1, Real, Device, Index > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkKokkos< 2, Real, Device, Index >
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
      // the auxiliary vector must have the same boundary values as the solution
      aux = ux;
      solver.init( { xSize, ySize } );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize ) const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize );
   }

   void
   exec( const Index xSize, const Index ySize ) override
   {
      solver.solve( ux, aux, { this->xDomainSize, this->yDomainSize }, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   VectorType ux, aux;
   MeanCurvatureFlowKokkosSolver< 2, Real, Device, Index > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkKokkos< 3, Real, Device, Index >
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
      // the auxiliary vector must have the same boundary values as the solution
      aux = ux;
      solver.init( { xSize, ySize, zSize } );
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
         aux,
         { this->xDomainSize, this->yDomainSize, this->zDomainSize },
         this->timeStep,
         this->finalTime,
         this->maxIterations );
   }

protected:
   VectorType ux, aux;
   MeanCurvatureFlowKokkosSolver< 3, Real, Device, Index > solver;
};

#endif
