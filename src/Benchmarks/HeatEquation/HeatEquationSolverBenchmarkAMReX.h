// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "HeatEquationSolverBenchmark.h"
#include "../GridLibraries/AMReXSupport.h"

#ifdef TNL_BENCHMARKS_HAVE_AMREX

   #include <algorithm>
   #include <array>
   #include <utility>

/**
 * \brief Heat equation solved with AMReX for the comparison with TNL.
 *
 * The finite difference and finite volume schemes are the same as in \ref HeatEquationSolverBenchmarkFDMParallelFor
 * and \ref HeatEquationSolverBenchmarkFVMParallelFor. The solution is stored in an array of AMReX boxes with one box
 * covering the whole grid, which is processed by `amrex::MFIter` with tiling and OpenMP on the CPU and by GPU
 * kernels without tiling on the GPU. In the finite volume scheme, the gradients are stored in face-centered arrays,
 * one for each face orientation. The initial condition is copied from the TNL vector initialized by the base
 * benchmark class and the solution is copied back only for writing the results.
 */
template< int Dimension, typename Real, typename Device, typename Index >
class HeatEquationAMReXSolver
{
public:
   using FabArray = TNL::Benchmarks::AMReXSupport::FabArray< Real >;
   using Sizes = std::array< int, Dimension >;
   using Steps = std::array< Real, Dimension >;

   template< typename Vector >
   void
   init( const Sizes& sizes, const Vector& u0, bool fvm )
   {
      this->sizes = sizes;
      domain = TNL::Benchmarks::AMReXSupport::createDomain< Dimension >( sizes );
      const amrex::BoxArray boxes( domain );
      const amrex::DistributionMapping distribution( boxes );
      u.define( boxes, distribution, 1, 0 );
      uNew.define( boxes, distribution, 1, 0 );
      // both arrays must have the boundary values of the initial condition
      TNL::Benchmarks::AMReXSupport::copy< Real, Dimension >( u0.getData(), u, sizes );
      TNL::Benchmarks::AMReXSupport::copy< Real, Dimension >( u0.getData(), uNew, sizes );
      if( fvm )
         for( int d = 0; d < Dimension; d++ )
            gradients[ d ].define( amrex::convert( boxes, amrex::IntVect::TheDimensionVector( d ) ), distribution, 1, 0 );
   }

   void
   solve( const Steps& domainSizes, Real timeStep, Real finalTime, Index maxIterations, bool fvm )
   {
      Steps h;
      Real hMin = domainSizes[ 0 ] / sizes[ 0 ];
      for( int d = 0; d < Dimension; d++ ) {
         h[ d ] = domainSizes[ d ] / sizes[ d ];
         hMin = std::min( hMin, h[ d ] );
      }
      const Real tau = timeStep ? timeStep : 0.1 * hMin * hMin;

      Real time = 0;
      Index iterations = 0;
      while( time < finalTime && ( ! maxIterations || iterations < maxIterations ) ) {
         if( fvm )
            fvmStep( u, uNew, h, tau );
         else
            fdmStep( u, uNew, h, tau );
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
   fdmStep( const FabArray& u, FabArray& uNew, const Steps& h, Real tau ) const
   {
      const amrex::Box interior = TNL::Benchmarks::AMReXSupport::getInterior< Dimension >( domain );
      const Real hx_inv = 1.0 / ( h[ 0 ] * h[ 0 ] );
      const Real hy_inv = Dimension > 1 ? 1.0 / ( h[ 1 ] * h[ 1 ] ) : 0.0;
      const Real hz_inv = Dimension > 2 ? 1.0 / ( h[ 2 ] * h[ 2 ] ) : 0.0;
   #ifdef AMREX_USE_OMP
      #pragma omp parallel if( amrex::Gpu::notInLaunchRegion() )
   #endif
      for( amrex::MFIter mfi( uNew, TNL::Benchmarks::AMReXSupport::getIterationInfo< Dimension >() ); mfi.isValid(); ++mfi ) {
         const amrex::Box box = mfi.tilebox() & interior;
         if( ! box.ok() )
            continue;
         const auto uArray = u.const_array( mfi );
         const auto uNewArray = uNew.array( mfi );
         // the conditions on the dimension are not `if constexpr`, because nvcc does not allow device lambdas to
         // capture a variable first in its branch
         amrex::ParallelFor(
            box,
            [ = ] AMREX_GPU_DEVICE( int i, int j, int k ) noexcept
            {
               const Real element = uArray( i, j, k );
               const Real center = (Real) 2.0 * element;
               Real laplace = ( uArray( i - 1, j, k ) - center + uArray( i + 1, j, k ) ) * hx_inv;
               if( Dimension > 1 )
                  laplace += ( uArray( i, j - 1, k ) - center + uArray( i, j + 1, k ) ) * hy_inv;
               if( Dimension > 2 )
                  laplace += ( uArray( i, j, k - 1 ) - center + uArray( i, j, k + 1 ) ) * hz_inv;
               uNewArray( i, j, k ) = element + laplace * tau;
            } );
      }
   }

   void
   fvmStep( const FabArray& u, FabArray& uNew, const Steps& h, Real tau )
   {
      Steps hInv;
      for( int d = 0; d < Dimension; d++ )
         hInv[ d ] = 1.0 / h[ d ];
      const amrex::Box interior = TNL::Benchmarks::AMReXSupport::getInterior< Dimension >( domain );

      // gradients on the faces needed by the interior cells: the face with index i along the axis d lies between
      // the cells i - 1 and i
   #ifdef AMREX_USE_OMP
      #pragma omp parallel if( amrex::Gpu::notInLaunchRegion() )
   #endif
      for( amrex::MFIter mfi( u, TNL::Benchmarks::AMReXSupport::getIterationInfo< Dimension >() ); mfi.isValid(); ++mfi ) {
         const auto uArray = u.const_array( mfi );
         for( int d = 0; d < Dimension; d++ ) {
            const amrex::Box box = mfi.nodaltilebox( d ) & amrex::surroundingNodes( interior, d );
            if( ! box.ok() )
               continue;
            const auto gradient = gradients[ d ].array( mfi );
            const int dx = d == 0;
            const int dy = d == 1;
            const int dz = d == 2;
            const Real hInv_d = hInv[ d ];
            amrex::ParallelFor(
               box,
               [ = ] AMREX_GPU_DEVICE( int i, int j, int k ) noexcept
               {
                  gradient( i, j, k ) = ( uArray( i, j, k ) - uArray( i - dx, j - dy, k - dz ) ) * hInv_d;
               } );
         }
      }

      // update of the interior cells
      const Real hx_inv = hInv[ 0 ];
      const Real hy_inv = Dimension > 1 ? hInv[ 1 ] : 0.0;
      const Real hz_inv = Dimension > 2 ? hInv[ 2 ] : 0.0;
   #ifdef AMREX_USE_OMP
      #pragma omp parallel if( amrex::Gpu::notInLaunchRegion() )
   #endif
      for( amrex::MFIter mfi( uNew, TNL::Benchmarks::AMReXSupport::getIterationInfo< Dimension >() ); mfi.isValid(); ++mfi ) {
         const amrex::Box box = mfi.tilebox() & interior;
         if( ! box.ok() )
            continue;
         const auto uArray = u.const_array( mfi );
         const auto uNewArray = uNew.array( mfi );
         const auto gx = gradients[ 0 ].const_array( mfi );
         const auto gy = gradients[ Dimension > 1 ? 1 : 0 ].const_array( mfi );
         const auto gz = gradients[ Dimension > 2 ? 2 : 0 ].const_array( mfi );
         amrex::ParallelFor(
            box,
            [ = ] AMREX_GPU_DEVICE( int i, int j, int k ) noexcept
            {
               Real divergence = ( gx( i + 1, j, k ) - gx( i, j, k ) ) * hx_inv;
               if( Dimension > 1 )
                  divergence += ( gy( i, j + 1, k ) - gy( i, j, k ) ) * hy_inv;
               if( Dimension > 2 )
                  divergence += ( gz( i, j, k + 1 ) - gz( i, j, k ) ) * hz_inv;
               uNewArray( i, j, k ) = uArray( i, j, k ) + tau * divergence;
            } );
      }
   }

protected:
   Sizes sizes;
   amrex::Box domain;
   FabArray u, uNew;
   std::array< FabArray, Dimension > gradients;
};

/**
 * \brief Heat equation benchmark with the AMReX implementation of the finite difference (`Fvm = false`) or the
 * finite volume (`Fvm = true`) scheme.
 */
template< int Dimension, bool Fvm, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmarkAMReX;

template< bool Fvm, typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkAMReX< 1, Fvm, Real, Device, Index >
: public HeatEquationSolverBenchmark< 1, Real, Device, Index >
{
   using BaseBenchmarkType = HeatEquationSolverBenchmark< 1, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;

   TNL::String
   scheme() override
   {
      return Fvm ? "fvm" : "fdm";
   }

   void
   init( const Index xSize ) override
   {
      BaseBenchmarkType::init( xSize, ux, aux );
      solver.init( { xSize }, ux, Fvm );
   }

   void
   getSolution( VectorType& u ) const override
   {
      solver.getSolution( ux );
      u = ux;
   }

   void
   exec( const Index xSize ) override
   {
      solver.solve( { this->xDomainSize }, this->timeStep, this->finalTime, this->maxIterations, Fvm );
   }

protected:
   mutable VectorType ux;
   VectorType aux;
   HeatEquationAMReXSolver< 1, Real, Device, Index > solver;
};

template< bool Fvm, typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkAMReX< 2, Fvm, Real, Device, Index >
: public HeatEquationSolverBenchmark< 2, Real, Device, Index >
{
   using BaseBenchmarkType = HeatEquationSolverBenchmark< 2, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;

   TNL::String
   scheme() override
   {
      return Fvm ? "fvm" : "fdm";
   }

   void
   init( const Index xSize, const Index ySize ) override
   {
      BaseBenchmarkType::init( xSize, ySize, ux, aux );
      solver.init( { xSize, ySize }, ux, Fvm );
   }

   void
   getSolution( VectorType& u ) const override
   {
      solver.getSolution( ux );
      u = ux;
   }

   void
   exec( const Index xSize, const Index ySize ) override
   {
      solver.solve( { this->xDomainSize, this->yDomainSize }, this->timeStep, this->finalTime, this->maxIterations, Fvm );
   }

protected:
   mutable VectorType ux;
   VectorType aux;
   HeatEquationAMReXSolver< 2, Real, Device, Index > solver;
};

template< bool Fvm, typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkAMReX< 3, Fvm, Real, Device, Index >
: public HeatEquationSolverBenchmark< 3, Real, Device, Index >
{
   using BaseBenchmarkType = HeatEquationSolverBenchmark< 3, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;

   TNL::String
   scheme() override
   {
      return Fvm ? "fvm" : "fdm";
   }

   void
   init( const Index xSize, const Index ySize, const Index zSize ) override
   {
      BaseBenchmarkType::init( xSize, ySize, zSize, ux, aux );
      solver.init( { xSize, ySize, zSize }, ux, Fvm );
   }

   void
   getSolution( VectorType& u ) const override
   {
      solver.getSolution( ux );
      u = ux;
   }

   void
   exec( const Index xSize, const Index ySize, const Index zSize ) override
   {
      solver.solve(
         { this->xDomainSize, this->yDomainSize, this->zDomainSize },
         this->timeStep,
         this->finalTime,
         this->maxIterations,
         Fvm );
   }

protected:
   mutable VectorType ux;
   VectorType aux;
   HeatEquationAMReXSolver< 3, Real, Device, Index > solver;
};

#endif
