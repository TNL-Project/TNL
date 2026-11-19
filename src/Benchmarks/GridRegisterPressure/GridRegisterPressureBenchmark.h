// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/staticFor.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Benchmarks/Devices.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL::Benchmarks::GridRegisterPressure {

//! \brief Number of rounds of the updates of the local variables in each cell.
constexpr int rounds = 4;

/**
 * \brief Computes the result in one cell from the values in the cell and in its six neighbours.
 *
 * The values initialize `Variables` local variables, which are then updated in \ref rounds rounds. Each update
 * depends on two other variables, so that all the variables stay alive during all rounds. `Variables` is a template
 * parameter and the loops are unrolled, so that the variables are stored in registers on the GPU.
 */
template< int Variables, typename Real >
__cuda_callable__
Real
compute( Real center, Real west, Real east, Real south, Real north, Real bottom, Real top )
{
   const Real values[ 7 ] = { center, west, east, south, north, bottom, top };
   Real v[ Variables ];
   Algorithms::staticFor< int, 0, Variables >(
      [ & ]( auto k )
      {
         v[ k ] = values[ k % 7 ] * (Real) ( 0.5 + 0.01 * k );
      } );
   Algorithms::staticFor< int, 0, rounds >(
      [ & ]( auto )
      {
         Algorithms::staticFor< int, 0, Variables >(
            [ & ]( auto k )
            {
               v[ k ] = (Real) 0.6 * v[ k ] + (Real) 0.3 * v[ ( k + 1 ) % Variables ] * v[ ( k + 7 ) % Variables ];
            } );
      } );
   Real result = 0;
   Algorithms::staticFor< int, 0, Variables >(
      [ & ]( auto k )
      {
         result += v[ k ] * (Real) ( 1.0 / ( k + 1 ) );
      } );
   return result;
}

/**
 * \brief Evaluates \ref compute in the interior cells of a 3D grid with `parallelFor` over plain multi-indices.
 */
template< int Variables, typename Real, typename Device, typename Index >
void
computeParallelFor(
   const Containers::Vector< Real, Device, Index >& u,
   Containers::Vector< Real, Device, Index >& result,
   Index n )
{
   using Coordinates = Containers::StaticArray< 3, Index >;
   const auto uView = u.getConstView();
   auto resultView = result.getView();
   const Index sy = n;
   const Index sz = n * n;
   auto kernel = [ = ] __cuda_callable__( const Coordinates& idx ) mutable
   {
      const Index i = ( idx[ 2 ] * n + idx[ 1 ] ) * n + idx[ 0 ];
      resultView[ i ] = compute< Variables >(
         uView[ i ], uView[ i - 1 ], uView[ i + 1 ], uView[ i - sy ], uView[ i + sy ], uView[ i - sz ], uView[ i + sz ] );
   };
   Algorithms::parallelFor< Device >( Coordinates{ 1, 1, 1 }, Coordinates{ n - 1, n - 1, n - 1 }, kernel );
}

/**
 * \brief Evaluates \ref compute in the interior cells of a 3D grid with the traversal of the grid cells.
 */
template< int Variables, typename Grid, typename Real, typename Device, typename Index >
void
computeGrid(
   const Grid& grid,
   const Containers::Vector< Real, Device, Index >& u,
   Containers::Vector< Real, Device, Index >& result )
{
   const auto uView = u.getConstView();
   auto resultView = result.getView();
   auto kernel = [ = ] __cuda_callable__( const typename Grid::Cell& cell ) mutable
   {
      const Index i = cell.getIndex();
      resultView[ i ] = compute< Variables >(
         uView[ i ],
         uView[ cell.template getAdjacentEntityIndex< 0, -1 >() ],
         uView[ cell.template getAdjacentEntityIndex< 0, 1 >() ],
         uView[ cell.template getAdjacentEntityIndex< 1, -1 >() ],
         uView[ cell.template getAdjacentEntityIndex< 1, 1 >() ],
         uView[ cell.template getAdjacentEntityIndex< 2, -1 >() ],
         uView[ cell.template getAdjacentEntityIndex< 2, 1 >() ] );
   };
   grid.template forInteriorEntities< 3 >( kernel );
}

/**
 * \brief Measures both implementations with the given number of local variables.
 */
template< int Variables, typename Real, typename Device >
void
runVariables( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   using Index = int;
   using Vector = Containers::Vector< Real, Device, Index >;
   using Grid = Meshes::Grid< 3, Real, Device, Index >;

   if( Variables > parameters.getParameter< int >( "max-variables" ) )
      return;
   const Index n = parameters.getParameter< int >( "size" );

   Containers::Vector< Real, Devices::Host, Index > hostU( n * n * n );
   for( Index i = 0; i < hostU.getSize(); i++ )
      hostU[ i ] = (Real) ( ( i * 2654435761u ) % 1000 ) / 1000;
   const Vector u( hostU );
   Vector resultParallelFor( u.getSize() );
   Vector resultGrid( u.getSize() );
   resultParallelFor = 0;
   resultGrid = 0;
   Grid grid;
   grid.setSizes( { n, n, n } );

   // both implementations must give the same results (up to the contraction of operations to FMA on GPUs)
   computeParallelFor< Variables >( u, resultParallelFor, n );
   computeGrid< Variables >( grid, u, resultGrid );
   const Real difference = max( abs( resultParallelFor - resultGrid ) );

   benchmark.setDatasetSize( 2 * u.getSize() * sizeof( Real ) );
   const auto run = [ & ]( const std::string& implementation, auto&& f )
   {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            { { "precision", getType< Real >() },
              { "size", convertToString( n ) },
              { "variables", convertToString( Variables ) },
              { "rounds", convertToString( rounds ) },
              { "implementation", implementation },
              { "max difference", convertToString( difference ) } } ) );
      benchmark.time< Device >( "TNL", f );
   };
   run(
      "parallel-for",
      [ & ]()
      {
         computeParallelFor< Variables >( u, resultParallelFor, n );
      } );
   run(
      "grid",
      [ & ]()
      {
         computeGrid< Variables >( grid, u, resultGrid );
      } );
}

template< typename Real, typename Device >
void
runBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   if( ! checkDevice< Device >( parameters ) )
      return;
   runVariables< 8, Real, Device >( benchmark, parameters );
   runVariables< 16, Real, Device >( benchmark, parameters );
   runVariables< 32, Real, Device >( benchmark, parameters );
   runVariables< 48, Real, Device >( benchmark, parameters );
   runVariables< 64, Real, Device >( benchmark, parameters );
   runVariables< 96, Real, Device >( benchmark, parameters );
   runVariables< 128, Real, Device >( benchmark, parameters );
}

}  // namespace TNL::Benchmarks::GridRegisterPressure
