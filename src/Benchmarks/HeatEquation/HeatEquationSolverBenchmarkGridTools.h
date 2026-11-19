// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "HeatEquationSolverBenchmark.h"
#include "../GridLibraries/GridToolsSupport.h"

#ifdef TNL_BENCHMARKS_HAVE_GRIDTOOLS

   #include <algorithm>
   #include <array>
   #include <type_traits>
   #include <utility>

   #include <TNL/Backend/Functions.h>

namespace HeatEquationGridTools {

namespace st = gridtools::stencil;
using TNL::Benchmarks::GridToolsSupport::Extent;
using TNL::Benchmarks::GridToolsSupport::unit;

/**
 * \brief Coefficients of the schemes passed to the stencils as a global parameter.
 *
 * `hInv` holds \f$ 1 / h_i^2 \f$ for the finite difference scheme and \f$ 1 / h_i \f$ for the finite volume scheme.
 */
template< typename Real >
struct Coefficients
{
   Real hInv[ 3 ];
   Real tau;
};

/**
 * \brief Stage of the finite difference scheme: `out = in + tau * laplace( in )` in the interior cells.
 */
template< int Dimension >
struct FDMStage
{
   using out = st::cartesian::inout_accessor< 0 >;
   using in = st::cartesian::in_accessor< 1, Extent< Dimension > >;
   using coefficients = st::cartesian::in_accessor< 2 >;
   using param_list = st::make_param_list< out, in, coefficients >;

   template< class Evaluation >
   GT_FUNCTION static void
   apply( Evaluation&& eval )
   {
      const auto c = eval( coefficients() );
      const auto element = eval( in() );
      using Real = std::decay_t< decltype( element ) >;
      const Real center = (Real) 2.0 * element;
      Real laplace = 0;
      for( int d = 0; d < Dimension; d++ )
         laplace += ( eval( in( -unit( d, 0 ), -unit( d, 1 ), -unit( d, 2 ) ) ) - center
                      + eval( in( unit( d, 0 ), unit( d, 1 ), unit( d, 2 ) ) ) )
                  * c.hInv[ d ];
      eval( out() ) = element + laplace * c.tau;
   }
};

/**
 * \brief Stage of the finite volume scheme computing the gradient on the faces orthogonal to the given axis.
 *
 * The face with the index `i` lies between the cells `i` and `i + 1` along the axis.
 */
template< int Axis >
struct FluxStage
{
   using flux = st::cartesian::inout_accessor< 0 >;
   using in = st::cartesian::in_accessor< 1, st::extent< 0, Axis == 0, 0, Axis == 1, 0, Axis == 2 > >;
   using coefficients = st::cartesian::in_accessor< 2 >;
   using param_list = st::make_param_list< flux, in, coefficients >;

   template< class Evaluation >
   GT_FUNCTION static void
   apply( Evaluation&& eval )
   {
      eval( flux() ) = ( eval( in( unit( Axis, 0 ), unit( Axis, 1 ), unit( Axis, 2 ) ) ) - eval( in() ) )
                     * eval( coefficients() ).hInv[ Axis ];
   }
};

/**
 * \brief Stage of the finite volume scheme updating the interior cells: `out = in + tau * div( flux )`.
 */
template< int Dimension >
struct FVMUpdateStage
{
   using out = st::cartesian::inout_accessor< 0 >;
   using in = st::cartesian::in_accessor< 1 >;
   using coefficients = st::cartesian::in_accessor< 2 >;
   using fx = st::cartesian::in_accessor< 3, st::extent< -1, 0 > >;
   using fy = st::cartesian::in_accessor< 4, st::extent< 0, 0, -1, 0 > >;
   using fz = st::cartesian::in_accessor< 5, st::extent< 0, 0, 0, 0, -1, 0 > >;
   using param_list = std::conditional_t<
      Dimension == 1,
      st::make_param_list< out, in, coefficients, fx >,
      std::conditional_t<
         Dimension == 2,
         st::make_param_list< out, in, coefficients, fx, fy >,
         st::make_param_list< out, in, coefficients, fx, fy, fz > > >;

   template< class Evaluation >
   GT_FUNCTION static void
   apply( Evaluation&& eval )
   {
      const auto c = eval( coefficients() );
      auto divergence = ( eval( fx() ) - eval( fx( -1, 0, 0 ) ) ) * c.hInv[ 0 ];
      if constexpr( Dimension > 1 )
         divergence += ( eval( fy() ) - eval( fy( 0, -1, 0 ) ) ) * c.hInv[ 1 ];
      if constexpr( Dimension > 2 )
         divergence += ( eval( fz() ) - eval( fz( 0, 0, -1 ) ) ) * c.hInv[ 2 ];
      eval( out() ) = eval( in() ) + c.tau * divergence;
   }
};

/**
 * \brief Composition of the finite volume stages, the gradients on the faces are GridTools temporaries.
 *
 * GridTools computes the temporaries on the domain extended by the offsets of the stages which read them.
 */
template< int Dimension, typename Real >
struct FVMSpecification
{
   template< class In, class Out, class Coefficients >
   auto
   operator()( In in, Out out, Coefficients coefficients ) const
   {
      if constexpr( Dimension == 1 ) {
         GT_DECLARE_TMP( Real, fx );
         return st::execute_parallel()
            .stage( FluxStage< 0 >(), fx, in, coefficients )
            .stage( FVMUpdateStage< 1 >(), out, in, coefficients, fx );
      }
      else if constexpr( Dimension == 2 ) {
         GT_DECLARE_TMP( Real, fx, fy );
         return st::execute_parallel()
            .stage( FluxStage< 0 >(), fx, in, coefficients )
            .stage( FluxStage< 1 >(), fy, in, coefficients )
            .stage( FVMUpdateStage< 2 >(), out, in, coefficients, fx, fy );
      }
      else {
         // In `execute_parallel`, the vertical levels are independent, so a stage must not read values computed by
         // a previous stage of the same pass on other levels. The update reads the flux `fz` on the level below,
         // hence the fluxes are computed in a separate pass. The horizontal offsets are synchronized by GridTools.
         GT_DECLARE_TMP( Real, fx, fy, fz );
         return st::multi_pass(
            st::execute_parallel()
               .stage( FluxStage< 0 >(), fx, in, coefficients )
               .stage( FluxStage< 1 >(), fy, in, coefficients )
               .stage( FluxStage< 2 >(), fz, in, coefficients ),
            st::execute_parallel().stage( FVMUpdateStage< 3 >(), out, in, coefficients, fx, fy, fz ) );
      }
   }
};

}  // namespace HeatEquationGridTools

/**
 * \brief Heat equation solved with GridTools for the comparison with TNL.
 *
 * The finite difference and finite volume schemes are the same as in \ref HeatEquationSolverBenchmarkFDMParallelFor
 * and \ref HeatEquationSolverBenchmarkFVMParallelFor. The data are stored in the TNL vectors initialized by the base
 * benchmark class and GridTools accesses them through synthetic SIDs. In the finite volume scheme, the gradients on
 * the faces are GridTools temporaries, which the library allocates and computes on the extended domain needed by the
 * update of the interior cells.
 */
template< int Dimension, typename Real, typename Device, typename Index >
class HeatEquationGridToolsSolver
{
public:
   using Backend = typename TNL::Benchmarks::GridToolsSupport::Backend< Device >::type;
   using Sizes = std::array< int, Dimension >;
   using Steps = std::array< Real, Dimension >;

   template< typename Vector >
   void
   solve(
      Vector& u,
      Vector& uNew,
      const Sizes& sizes,
      const Steps& domainSizes,
      Real timeStep,
      Real finalTime,
      Index maxIterations,
      bool fvm ) const
   {
      namespace st = gridtools::stencil;
      Steps h;
      Real hMin = domainSizes[ 0 ] / sizes[ 0 ];
      for( int d = 0; d < Dimension; d++ ) {
         h[ d ] = domainSizes[ d ] / sizes[ d ];
         hMin = std::min( hMin, h[ d ] );
      }
      HeatEquationGridTools::Coefficients< Real > coefficients{};
      coefficients.tau = timeStep ? timeStep : 0.1 * hMin * hMin;
      for( int d = 0; d < Dimension; d++ )
         coefficients.hInv[ d ] = fvm ? 1.0 / h[ d ] : 1.0 / ( h[ d ] * h[ d ] );

      const TNL::Benchmarks::GridToolsSupport::Interior< Dimension > interior( sizes );
      const auto grid = interior.getGrid();
      const st::global_parameter coefficientsParameter( coefficients );
      const auto fdm = []( auto in, auto out, auto coefficients )
      {
         return st::execute_parallel().stage( HeatEquationGridTools::FDMStage< Dimension >(), out, in, coefficients );
      };
      const HeatEquationGridTools::FVMSpecification< Dimension, Real > fvmSpecification;

      Real* uData = u.getData();
      Real* uNewData = uNew.getData();
      Real time = 0;
      Index iterations = 0;
      while( time < finalTime && ( ! maxIterations || iterations < maxIterations ) ) {
         if( fvm )
            st::run(
               fvmSpecification, Backend(), grid, interior.wrap( uData ), interior.wrap( uNewData ), coefficientsParameter );
         else
            st::run( fdm, Backend(), grid, interior.wrap( uData ), interior.wrap( uNewData ), coefficientsParameter );
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
};

/**
 * \brief Heat equation benchmark with the GridTools implementation of the finite difference (`Fvm = false`) or the
 * finite volume (`Fvm = true`) scheme.
 */
template< int Dimension, bool Fvm, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmarkGridTools;

template< bool Fvm, typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkGridTools< 1, Fvm, Real, Device, Index >
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
   }

   void
   getSolution( VectorType& u ) const override
   {
      u = this->ux;
   }

   void
   exec( const Index xSize ) override
   {
      solver.solve( ux, aux, { xSize }, { this->xDomainSize }, this->timeStep, this->finalTime, this->maxIterations, Fvm );
   }

protected:
   VectorType ux, aux;
   HeatEquationGridToolsSolver< 1, Real, Device, Index > solver;
};

template< bool Fvm, typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkGridTools< 2, Fvm, Real, Device, Index >
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
         ux,
         aux,
         { xSize, ySize },
         { this->xDomainSize, this->yDomainSize },
         this->timeStep,
         this->finalTime,
         this->maxIterations,
         Fvm );
   }

protected:
   VectorType ux, aux;
   HeatEquationGridToolsSolver< 2, Real, Device, Index > solver;
};

template< bool Fvm, typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkGridTools< 3, Fvm, Real, Device, Index >
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
         { xSize, ySize, zSize },
         { this->xDomainSize, this->yDomainSize, this->zDomainSize },
         this->timeStep,
         this->finalTime,
         this->maxIterations,
         Fvm );
   }

protected:
   VectorType ux, aux;
   HeatEquationGridToolsSolver< 3, Real, Device, Index > solver;
};

#endif
