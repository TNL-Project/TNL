// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "HeatEquationSolverBenchmark.h"
#include "../GridLibraries/KokkosSupport.h"

#ifdef TNL_BENCHMARKS_HAVE_KOKKOS

   #include <algorithm>
   #include <array>
   #include <utility>

/**
 * \brief Heat equation solved with Kokkos for the comparison with TNL.
 *
 * The finite difference and finite volume schemes are the same as in \ref HeatEquationSolverBenchmarkFDMParallelFor
 * and \ref HeatEquationSolverBenchmarkFVMParallelFor. The data are stored in the TNL vectors initialized by the base
 * benchmark class and Kokkos accesses them through unmanaged views. In the finite volume scheme, the gradients on
 * the faces are stored in separate Kokkos views, one for each face orientation.
 */
template< int Dimension, typename Real, typename Device, typename Index >
class HeatEquationKokkosSolver
{
public:
   using ExecSpace = typename TNL::Benchmarks::KokkosSupport::ExecutionSpace< Device >::type;
   using View = TNL::Benchmarks::KokkosSupport::View< Real, Dimension, ExecSpace >;
   using UnmanagedView = TNL::Benchmarks::KokkosSupport::UnmanagedView< Real, Dimension, ExecSpace >;
   using Sizes = std::array< int, Dimension >;
   using Steps = std::array< Real, Dimension >;

   void
   init( const Sizes& sizes, bool fvm )
   {
      this->sizes = sizes;
      if( fvm )
         for( int d = 0; d < Dimension; d++ ) {
            Sizes faceSizes = sizes;
            faceSizes[ d ]++;
            gradients[ d ] = TNL::Benchmarks::KokkosSupport::createView< View, Dimension >( "gradients", faceSizes );
         }
   }

   template< typename Vector >
   void
   solve( Vector& u, Vector& uNew, const Steps& domainSizes, Real timeStep, Real finalTime, Index maxIterations, bool fvm )
   {
      Steps h;
      Real hMin = domainSizes[ 0 ] / sizes[ 0 ];
      for( int d = 0; d < Dimension; d++ ) {
         h[ d ] = domainSizes[ d ] / sizes[ d ];
         hMin = std::min( hMin, h[ d ] );
      }
      const Real tau = timeStep ? timeStep : 0.1 * hMin * hMin;

      auto uView = TNL::Benchmarks::KokkosSupport::wrap< UnmanagedView, Dimension >( u.getData(), sizes );
      auto uNewView = TNL::Benchmarks::KokkosSupport::wrap< UnmanagedView, Dimension >( uNew.getData(), sizes );
      Real time = 0;
      Index iterations = 0;
      while( time < finalTime && ( ! maxIterations || iterations < maxIterations ) ) {
         if( fvm )
            fvmStep( uView, uNewView, h, tau );
         else
            fdmStep( uView, uNewView, h, tau );
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
   fdmStep( const UnmanagedView& u, const UnmanagedView& uNew, const Steps& h, Real tau ) const
   {
      const auto [ begin, end ] = TNL::Benchmarks::KokkosSupport::getInterior< Dimension >( sizes );
      const Real hx_inv = 1.0 / ( h[ 0 ] * h[ 0 ] );
      if constexpr( Dimension == 1 ) {
         TNL::Benchmarks::KokkosSupport::parallelFor< ExecSpace, Dimension >(
            "heat-equation-fdm-1d", begin, end, KOKKOS_LAMBDA( int i ) {
               const Real element = u( i );
               const Real center = (Real) 2.0 * element;
               uNew( i ) = element + ( ( u( i - 1 ) - center + u( i + 1 ) ) * hx_inv ) * tau;
            } );
      }
      else if constexpr( Dimension == 2 ) {
         const Real hy_inv = 1.0 / ( h[ 1 ] * h[ 1 ] );
         TNL::Benchmarks::KokkosSupport::parallelFor< ExecSpace, Dimension >(
            "heat-equation-fdm-2d", begin, end, KOKKOS_LAMBDA( int i, int j ) {
               const Real element = u( i, j );
               const Real center = (Real) 2.0 * element;
               uNew( i, j ) = element
                            + ( ( u( i - 1, j ) - center + u( i + 1, j ) ) * hx_inv
                                + ( u( i, j - 1 ) - center + u( i, j + 1 ) ) * hy_inv )
                                 * tau;
            } );
      }
      else {
         const Real hy_inv = 1.0 / ( h[ 1 ] * h[ 1 ] );
         const Real hz_inv = 1.0 / ( h[ 2 ] * h[ 2 ] );
         TNL::Benchmarks::KokkosSupport::parallelFor< ExecSpace, Dimension >(
            "heat-equation-fdm-3d", begin, end, KOKKOS_LAMBDA( int i, int j, int k ) {
               const Real element = u( i, j, k );
               const Real center = (Real) 2.0 * element;
               uNew( i, j, k ) = element
                               + ( ( u( i - 1, j, k ) - center + u( i + 1, j, k ) ) * hx_inv
                                   + ( u( i, j - 1, k ) - center + u( i, j + 1, k ) ) * hy_inv
                                   + ( u( i, j, k - 1 ) - center + u( i, j, k + 1 ) ) * hz_inv )
                                    * tau;
            } );
      }
   }

   void
   fvmStep( const UnmanagedView& u, const UnmanagedView& uNew, const Steps& h, Real tau ) const
   {
      Steps hInv;
      for( int d = 0; d < Dimension; d++ )
         hInv[ d ] = 1.0 / h[ d ];

      // gradients on the faces needed by the interior cells: the faces orthogonal to the axis d are indexed
      // by the index of the cell which follows the face along the axis d
      const auto [ interiorBegin, interiorEnd ] = TNL::Benchmarks::KokkosSupport::getInterior< Dimension >( sizes );
      for( int d = 0; d < Dimension; d++ ) {
         Sizes end = interiorEnd;
         end[ d ] = sizes[ d ];
         const View gradient = gradients[ d ];
         const Real hInv_d = hInv[ d ];
         if constexpr( Dimension == 1 ) {
            TNL::Benchmarks::KokkosSupport::parallelFor< ExecSpace, Dimension >(
               "heat-equation-fvm-gradients-1d", interiorBegin, end, KOKKOS_LAMBDA( int i ) {
                  gradient( i ) = ( u( i ) - u( i - 1 ) ) * hInv_d;
               } );
         }
         else if constexpr( Dimension == 2 ) {
            const int dx = d == 0;
            const int dy = d == 1;
            TNL::Benchmarks::KokkosSupport::parallelFor< ExecSpace, Dimension >(
               "heat-equation-fvm-gradients-2d", interiorBegin, end, KOKKOS_LAMBDA( int i, int j ) {
                  gradient( i, j ) = ( u( i, j ) - u( i - dx, j - dy ) ) * hInv_d;
               } );
         }
         else {
            const int dx = d == 0;
            const int dy = d == 1;
            const int dz = d == 2;
            TNL::Benchmarks::KokkosSupport::parallelFor< ExecSpace, Dimension >(
               "heat-equation-fvm-gradients-3d", interiorBegin, end, KOKKOS_LAMBDA( int i, int j, int k ) {
                  gradient( i, j, k ) = ( u( i, j, k ) - u( i - dx, j - dy, k - dz ) ) * hInv_d;
               } );
         }
      }

      // update of the interior cells
      if constexpr( Dimension == 1 ) {
         const View gx = gradients[ 0 ];
         const Real hx_inv = hInv[ 0 ];
         TNL::Benchmarks::KokkosSupport::parallelFor< ExecSpace, Dimension >(
            "heat-equation-fvm-update-1d", interiorBegin, interiorEnd, KOKKOS_LAMBDA( int i ) {
               uNew( i ) = u( i ) + tau * ( gx( i + 1 ) - gx( i ) ) * hx_inv;
            } );
      }
      else if constexpr( Dimension == 2 ) {
         const View gx = gradients[ 0 ];
         const View gy = gradients[ 1 ];
         const Real hx_inv = hInv[ 0 ];
         const Real hy_inv = hInv[ 1 ];
         TNL::Benchmarks::KokkosSupport::parallelFor< ExecSpace, Dimension >(
            "heat-equation-fvm-update-2d", interiorBegin, interiorEnd, KOKKOS_LAMBDA( int i, int j ) {
               uNew( i, j ) =
                  u( i, j ) + tau * ( ( gx( i + 1, j ) - gx( i, j ) ) * hx_inv + ( gy( i, j + 1 ) - gy( i, j ) ) * hy_inv );
            } );
      }
      else {
         const View gx = gradients[ 0 ];
         const View gy = gradients[ 1 ];
         const View gz = gradients[ 2 ];
         const Real hx_inv = hInv[ 0 ];
         const Real hy_inv = hInv[ 1 ];
         const Real hz_inv = hInv[ 2 ];
         TNL::Benchmarks::KokkosSupport::parallelFor< ExecSpace, Dimension >(
            "heat-equation-fvm-update-3d", interiorBegin, interiorEnd, KOKKOS_LAMBDA( int i, int j, int k ) {
               uNew( i, j, k ) =
                  u( i, j, k )
                  + tau
                       * ( ( gx( i + 1, j, k ) - gx( i, j, k ) ) * hx_inv + ( gy( i, j + 1, k ) - gy( i, j, k ) ) * hy_inv
                           + ( gz( i, j, k + 1 ) - gz( i, j, k ) ) * hz_inv );
            } );
      }
   }

protected:
   Sizes sizes;
   std::array< View, Dimension > gradients;
};

/**
 * \brief Heat equation benchmark with the Kokkos implementation of the finite difference (`Fvm = false`) or the
 * finite volume (`Fvm = true`) scheme.
 */
template< int Dimension, bool Fvm, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmarkKokkos;

template< bool Fvm, typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkKokkos< 1, Fvm, Real, Device, Index >
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
      solver.init( { xSize }, Fvm );
   }

   void
   getSolution( VectorType& u ) const override
   {
      u = this->ux;
   }

   void
   exec( const Index xSize ) override
   {
      solver.solve( ux, aux, { this->xDomainSize }, this->timeStep, this->finalTime, this->maxIterations, Fvm );
   }

protected:
   VectorType ux, aux;
   HeatEquationKokkosSolver< 1, Real, Device, Index > solver;
};

template< bool Fvm, typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkKokkos< 2, Fvm, Real, Device, Index >
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
      solver.init( { xSize, ySize }, Fvm );
   }

   void
   getSolution( VectorType& u ) const override
   {
      u = this->ux;
   }

   void
   exec( const Index xSize, const Index ySize ) override
   {
      solver.solve(
         ux, aux, { this->xDomainSize, this->yDomainSize }, this->timeStep, this->finalTime, this->maxIterations, Fvm );
   }

protected:
   VectorType ux, aux;
   HeatEquationKokkosSolver< 2, Real, Device, Index > solver;
};

template< bool Fvm, typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkKokkos< 3, Fvm, Real, Device, Index >
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
      solver.init( { xSize, ySize, zSize }, Fvm );
   }

   void
   getSolution( VectorType& u ) const override
   {
      u = this->ux;
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
         this->maxIterations,
         Fvm );
   }

protected:
   VectorType ux, aux;
   HeatEquationKokkosSolver< 3, Real, Device, Index > solver;
};

#endif
