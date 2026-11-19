// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "MeanCurvatureFlowSolverBenchmark.h"
#include "../GridLibraries/AMReXSupport.h"

#ifdef TNL_BENCHMARKS_HAVE_AMREX

   #include <algorithm>
   #include <array>
   #include <utility>

/**
 * \brief Mean-curvature flow solved with AMReX for the comparison with TNL.
 *
 * The finite difference scheme and the explicit Euler method are the same as in \ref MeanCurvatureFlowFDMGrid. The
 * solution and the approximations \f$q\f$ and \f$\bar q\f$ of the gradient norm are stored in arrays of AMReX boxes
 * with one box covering the whole grid, which is processed by `amrex::MFIter` with tiling and OpenMP on the CPU and
 * by GPU kernels without tiling on the GPU. The initial condition is copied from the TNL vector initialized by the
 * base benchmark class and the solution is copied back only for writing the results.
 */
template< int Dimension, typename Real, typename Device, typename Index >
class MeanCurvatureFlowAMReXSolver
{
public:
   using FabArray = TNL::Benchmarks::AMReXSupport::FabArray< Real >;
   using Sizes = std::array< int, Dimension >;
   using Steps = std::array< Real, Dimension >;

   template< typename Vector >
   void
   init( const Sizes& sizes, const Vector& u0 )
   {
      this->sizes = sizes;
      domain = TNL::Benchmarks::AMReXSupport::createDomain< Dimension >( sizes );
      const amrex::BoxArray boxes( domain );
      const amrex::DistributionMapping distribution( boxes );
      u.define( boxes, distribution, 1, 0 );
      uNew.define( boxes, distribution, 1, 0 );
      q.define( boxes, distribution, 1, 0 );
      qBar.define( boxes, distribution, 1, 0 );
      // both arrays must have the boundary values of the initial condition
      TNL::Benchmarks::AMReXSupport::copy< Real, Dimension >( u0.getData(), u, sizes );
      TNL::Benchmarks::AMReXSupport::copy< Real, Dimension >( u0.getData(), uNew, sizes );
      // the values on the boundary are not computed
      q.setVal( 1.0 );
      qBar.setVal( 1.0 );
   }

   void
   solve( const Steps& domainSizes, Real timeStep, Real finalTime, Index maxIterations )
   {
      Steps h;
      Real hMin = domainSizes[ 0 ] / sizes[ 0 ];
      for( int d = 0; d < Dimension; d++ ) {
         h[ d ] = domainSizes[ d ] / sizes[ d ];
         hMin = std::min( hMin, h[ d ] );
      }
      const Real tau = timeStep > 0 ? timeStep : 0.1 * hMin * hMin;

      Real time = 0;
      Index iterations = 0;
      while( time < finalTime && ( ! maxIterations || iterations < maxIterations ) ) {
         computeGradientNorms( u, h );
         explicitStep( u, uNew, h, tau );
         std::swap( u, uNew );
         time += tau;
         iterations++;
      }
      amrex::Gpu::streamSynchronize();
   }

   template< typename Vector >
   void
   getSolution( Vector& target ) const
   {
      TNL::Benchmarks::AMReXSupport::copy< Real, Dimension >( u, target.getData(), sizes );
   }

   void
   computeGradientNorms( const FabArray& u, const Steps& h )
   {
      amrex::GpuArray< Real, Dimension > hInv;
      for( int d = 0; d < Dimension; d++ )
         hInv[ d ] = 1.0 / h[ d ];
      const amrex::Box interior = TNL::Benchmarks::AMReXSupport::getInterior< Dimension >( domain );
   #ifdef AMREX_USE_OMP
      #pragma omp parallel if( amrex::Gpu::notInLaunchRegion() )
   #endif
      for( amrex::MFIter mfi( q, TNL::Benchmarks::AMReXSupport::getIterationInfo< Dimension >() ); mfi.isValid(); ++mfi ) {
         const amrex::Box box = mfi.tilebox() & interior;
         if( ! box.ok() )
            continue;
         const auto uArray = u.const_array( mfi );
         const auto qArray = q.array( mfi );
         const auto qBarArray = qBar.array( mfi );
         amrex::ParallelFor(
            box,
            [ = ] AMREX_GPU_DEVICE( int i, int j, int k ) noexcept
            {
               const amrex::IntVect c( i, j, k );
               const Real u_c = uArray( c );
               Real forwardSum = 0;
               Real backwardSum = 0;
               for( int d = 0; d < Dimension; d++ ) {
                  const amrex::IntVect e = amrex::IntVect::TheDimensionVector( d );
                  const Real forward = ( uArray( c + e ) - u_c ) * hInv[ d ];
                  const Real backward = ( u_c - uArray( c - e ) ) * hInv[ d ];
                  forwardSum += forward * forward;
                  backwardSum += backward * backward;
               }
               qArray( c ) = std::sqrt( 1.0 + forwardSum );
               qBarArray( c ) = std::sqrt( 1.0 + 0.5 * ( forwardSum + backwardSum ) );
            } );
      }
   }

   void
   explicitStep( const FabArray& u, FabArray& uNew, const Steps& h, Real tau ) const
   {
      amrex::GpuArray< Real, Dimension > hSqrInv;
      for( int d = 0; d < Dimension; d++ )
         hSqrInv[ d ] = 1.0 / ( h[ d ] * h[ d ] );
      const amrex::Box interior = TNL::Benchmarks::AMReXSupport::getInterior< Dimension >( domain );
   #ifdef AMREX_USE_OMP
      #pragma omp parallel if( amrex::Gpu::notInLaunchRegion() )
   #endif
      for( amrex::MFIter mfi( uNew, TNL::Benchmarks::AMReXSupport::getIterationInfo< Dimension >() ); mfi.isValid(); ++mfi ) {
         const amrex::Box box = mfi.tilebox() & interior;
         if( ! box.ok() )
            continue;
         const auto uArray = u.const_array( mfi );
         const auto qArray = q.const_array( mfi );
         const auto qBarArray = qBar.const_array( mfi );
         const auto uNewArray = uNew.array( mfi );
         amrex::ParallelFor(
            box,
            [ = ] AMREX_GPU_DEVICE( int i, int j, int k ) noexcept
            {
               const amrex::IntVect c( i, j, k );
               const Real u_c = uArray( c );
               const Real q_c = qArray( c );
               Real divergence = 0;
               for( int d = 0; d < Dimension; d++ ) {
                  const amrex::IntVect e = amrex::IntVect::TheDimensionVector( d );
                  divergence +=
                     ( ( uArray( c + e ) - u_c ) / q_c - ( u_c - uArray( c - e ) ) / qArray( c - e ) ) * hSqrInv[ d ];
               }
               uNewArray( c ) = u_c + tau * qBarArray( c ) * divergence;
            } );
      }
   }

protected:
   Sizes sizes;
   amrex::Box domain;
   FabArray u, uNew, q, qBar;
};

/**
 * \brief Mean-curvature flow benchmark with the AMReX implementation of the explicit finite difference scheme.
 */
template< int Dimension, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct MeanCurvatureFlowSolverBenchmarkAMReX;

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkAMReX< 1, Real, Device, Index >
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
      solver.getSolution( ux );
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

   void
   exec( const Index xSize ) override
   {
      solver.solve( { this->xDomainSize }, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   mutable VectorType ux;
   VectorType aux;
   MeanCurvatureFlowAMReXSolver< 1, Real, Device, Index > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkAMReX< 2, Real, Device, Index >
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
      solver.getSolution( ux );
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize );
   }

   void
   exec( const Index xSize, const Index ySize ) override
   {
      solver.solve( { this->xDomainSize, this->yDomainSize }, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   mutable VectorType ux;
   VectorType aux;
   MeanCurvatureFlowAMReXSolver< 2, Real, Device, Index > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkAMReX< 3, Real, Device, Index >
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
      solver.getSolution( ux );
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize, zSize, zSlice );
   }

   void
   exec( const Index xSize, const Index ySize, const Index zSize ) override
   {
      solver.solve(
         { this->xDomainSize, this->yDomainSize, this->zDomainSize }, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   mutable VectorType ux;
   VectorType aux;
   MeanCurvatureFlowAMReXSolver< 3, Real, Device, Index > solver;
};

#endif
