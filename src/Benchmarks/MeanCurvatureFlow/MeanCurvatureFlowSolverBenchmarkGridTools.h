// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "MeanCurvatureFlowSolverBenchmark.h"
#include "../GridLibraries/GridToolsSupport.h"

#ifdef TNL_BENCHMARKS_HAVE_GRIDTOOLS

   #include <algorithm>
   #include <array>
   #include <type_traits>
   #include <utility>

   #include <gridtools/common/gt_math.hpp>

   #include <TNL/Backend/Functions.h>

namespace MeanCurvatureFlowGridTools {

namespace st = gridtools::stencil;
using TNL::Benchmarks::GridToolsSupport::Extent;
using TNL::Benchmarks::GridToolsSupport::unit;

/**
 * \brief Coefficients of the scheme passed to the stencils as a global parameter.
 */
template< typename Real >
struct Coefficients
{
   Real hInv[ 3 ];
   Real hSqrInv[ 3 ];
   Real tau;
};

/**
 * \brief Stage computing the approximations \f$q\f$ and \f$\bar q\f$ of the gradient norm in the interior cells.
 */
template< int Dimension >
struct GradientNormsStage
{
   using q = st::cartesian::inout_accessor< 0 >;
   using qBar = st::cartesian::inout_accessor< 1 >;
   using u = st::cartesian::in_accessor< 2, Extent< Dimension > >;
   using coefficients = st::cartesian::in_accessor< 3 >;
   using param_list = st::make_param_list< q, qBar, u, coefficients >;

   template< class Evaluation >
   GT_FUNCTION static void
   apply( Evaluation&& eval )
   {
      const auto c = eval( coefficients() );
      const auto u_c = eval( u() );
      using Real = std::decay_t< decltype( u_c ) >;
      Real forwardSum = 0;
      Real backwardSum = 0;
      for( int i = 0; i < Dimension; i++ ) {
         const Real forward = ( eval( u( unit( i, 0 ), unit( i, 1 ), unit( i, 2 ) ) ) - u_c ) * c.hInv[ i ];
         const Real backward = ( u_c - eval( u( -unit( i, 0 ), -unit( i, 1 ), -unit( i, 2 ) ) ) ) * c.hInv[ i ];
         forwardSum += forward * forward;
         backwardSum += backward * backward;
      }
      eval( q() ) = gridtools::math::sqrt( 1.0 + forwardSum );
      eval( qBar() ) = gridtools::math::sqrt( 1.0 + 0.5 * ( forwardSum + backwardSum ) );
   }
};

/**
 * \brief Stage of the explicit Euler method: `uNew = u + tau * operator( u )` in the interior cells.
 */
template< int Dimension >
struct ExplicitStepStage
{
   using uNew = st::cartesian::inout_accessor< 0 >;
   using u = st::cartesian::in_accessor< 1, Extent< Dimension > >;
   using q = st::cartesian::in_accessor< 2, Extent< Dimension, 1, 0 > >;
   using qBar = st::cartesian::in_accessor< 3 >;
   using coefficients = st::cartesian::in_accessor< 4 >;
   using param_list = st::make_param_list< uNew, u, q, qBar, coefficients >;

   template< class Evaluation >
   GT_FUNCTION static void
   apply( Evaluation&& eval )
   {
      const auto c = eval( coefficients() );
      const auto u_c = eval( u() );
      const auto q_c = eval( q() );
      using Real = std::decay_t< decltype( u_c ) >;
      Real divergence = 0;
      for( int i = 0; i < Dimension; i++ ) {
         const int dx = unit( i, 0 );
         const int dy = unit( i, 1 );
         const int dz = unit( i, 2 );
         divergence +=
            ( ( eval( u( dx, dy, dz ) ) - u_c ) / q_c - ( u_c - eval( u( -dx, -dy, -dz ) ) ) / eval( q( -dx, -dy, -dz ) ) )
            * c.hSqrInv[ i ];
      }
      eval( uNew() ) = u_c + c.tau * eval( qBar() ) * divergence;
   }
};

}  // namespace MeanCurvatureFlowGridTools

/**
 * \brief Mean-curvature flow solved with GridTools for the comparison with TNL.
 *
 * The finite difference scheme and the explicit Euler method are the same as in \ref MeanCurvatureFlowFDMGrid. The
 * solution and the approximations \f$q\f$ and \f$\bar q\f$ of the gradient norm are stored in TNL vectors and
 * GridTools accesses them through synthetic SIDs. The two stages are run as separate computations, because GridTools
 * would otherwise extend the computation of \f$q\f$ to the boundary cells, where the scheme keeps \f$q = 1\f$.
 */
template< int Dimension, typename Real, typename Device, typename Index >
class MeanCurvatureFlowGridToolsSolver
{
public:
   using Backend = typename TNL::Benchmarks::GridToolsSupport::Backend< Device >::type;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using Sizes = std::array< int, Dimension >;
   using Steps = std::array< Real, Dimension >;

   void
   init( Index size )
   {
      // the values on the boundary are not computed
      q.setSize( size );
      qBar.setSize( size );
      q = 1.0;
      qBar = 1.0;
   }

   void
   solve(
      VectorType& u,
      VectorType& uNew,
      const Sizes& sizes,
      const Steps& domainSizes,
      Real timeStep,
      Real finalTime,
      Index maxIterations )
   {
      namespace st = gridtools::stencil;
      using namespace MeanCurvatureFlowGridTools;
      Steps h;
      Real hMin = domainSizes[ 0 ] / sizes[ 0 ];
      for( int d = 0; d < Dimension; d++ ) {
         h[ d ] = domainSizes[ d ] / sizes[ d ];
         hMin = std::min( hMin, h[ d ] );
      }
      Coefficients< Real > coefficients{};
      coefficients.tau = timeStep > 0 ? timeStep : 0.1 * hMin * hMin;
      for( int d = 0; d < Dimension; d++ ) {
         coefficients.hInv[ d ] = 1.0 / h[ d ];
         coefficients.hSqrInv[ d ] = 1.0 / ( h[ d ] * h[ d ] );
      }

      const TNL::Benchmarks::GridToolsSupport::Interior< Dimension > interior( sizes );
      const auto grid = interior.getGrid();
      const st::global_parameter coefficientsParameter( coefficients );
      const auto gradientNorms = []( auto u, auto q, auto qBar, auto coefficients )
      {
         return st::execute_parallel().stage( GradientNormsStage< Dimension >(), q, qBar, u, coefficients );
      };
      const auto explicitStep = []( auto u, auto q, auto qBar, auto uNew, auto coefficients )
      {
         return st::execute_parallel().stage( ExplicitStepStage< Dimension >(), uNew, u, q, qBar, coefficients );
      };
      const auto qSid = interior.wrap( q.getData() );
      const auto qBarSid = interior.wrap( qBar.getData() );

      Real* uData = u.getData();
      Real* uNewData = uNew.getData();
      Real time = 0;
      Index iterations = 0;
      while( time < finalTime && ( ! maxIterations || iterations < maxIterations ) ) {
         const auto uSid = interior.wrap( uData );
         st::run( gradientNorms, Backend(), grid, uSid, qSid, qBarSid, coefficientsParameter );
         st::run( explicitStep, Backend(), grid, uSid, qSid, qBarSid, interior.wrap( uNewData ), coefficientsParameter );
         std::swap( uData, uNewData );
         time += coefficients.tau;
         iterations++;
      }
      if constexpr( std::is_same_v< Device, TNL::Devices::GPU > )
         TNL::Backend::deviceSynchronize();
      // the TNL vectors must hold the solution in the same vector as in the other implementations
      if( iterations % 2 == 1 )
         u.swap( uNew );
   }

protected:
   VectorType q, qBar;
};

/**
 * \brief Mean-curvature flow benchmark with the GridTools implementation of the explicit finite difference scheme.
 */
template< int Dimension, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct MeanCurvatureFlowSolverBenchmarkGridTools;

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkGridTools< 1, Real, Device, Index >
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
      solver.init( ux.getSize() );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize ) const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

   void
   exec( const Index xSize ) override
   {
      solver.solve( ux, aux, { xSize }, { this->xDomainSize }, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   VectorType ux, aux;
   MeanCurvatureFlowGridToolsSolver< 1, Real, Device, Index > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkGridTools< 2, Real, Device, Index >
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
      solver.init( ux.getSize() );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize ) const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize );
   }

   void
   exec( const Index xSize, const Index ySize ) override
   {
      solver.solve(
         ux,
         aux,
         { xSize, ySize },
         { this->xDomainSize, this->yDomainSize },
         this->timeStep,
         this->finalTime,
         this->maxIterations );
   }

protected:
   VectorType ux, aux;
   MeanCurvatureFlowGridToolsSolver< 2, Real, Device, Index > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowSolverBenchmarkGridTools< 3, Real, Device, Index >
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
      solver.init( ux.getSize() );
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
         { xSize, ySize, zSize },
         { this->xDomainSize, this->yDomainSize, this->zDomainSize },
         this->timeStep,
         this->finalTime,
         this->maxIterations );
   }

protected:
   VectorType ux, aux;
   MeanCurvatureFlowGridToolsSolver< 3, Real, Device, Index > solver;
};

#endif
