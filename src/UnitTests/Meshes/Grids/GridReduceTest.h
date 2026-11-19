#pragma once

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/reduce.h>
#include <TNL/Meshes/traverse.h>

// Value of the entity with the given index. The values are small integers, so that their sums are exact
// in floating-point arithmetic regardless of the order of the operations.
template< typename Real, typename Index >
__cuda_callable__
Real
entityValue( Index index )
{
   return ( index * 7 ) % 11 - 3;
}

// Accumulates the expected results of the reductions in forEntities on the host. It is a class instead of
// a lambda function, because nvcc does not allow local types in the template arguments of the functions
// with extended lambda functions.
template< typename Real, typename Index >
struct ExpectedResults
{
   Real* sum;
   Real* max;
   Index* count;

   template< typename Entity >
   __cuda_callable__
   void
   operator()( const Entity& entity ) const
   {
      const Real value = entityValue< Real >( entity.getIndex() );
      *sum += value;
      *max = TNL::max( *max, value );
      ( *count )++;
   }
};

// Checks the reductions over the region [ begin, end ) against the results of forEntities on the host.
template< int EntityDimension, typename Grid >
void
testReduceRegion(
   const Grid& grid,
   const typename Grid::CoordinatesType& begin,
   const typename Grid::CoordinatesType& end,
   bool wholeGrid )
{
   using Real = typename Grid::RealType;
   using Device = typename Grid::DeviceType;
   using Index = typename Grid::IndexType;
   using Entity = typename Grid::template EntityType< EntityDimension >;
   using HostGrid = TNL::Meshes::Grid< Grid::getMeshDimension(), Real, TNL::Devices::Sequential, Index >;
   SCOPED_TRACE(
      "Entity dimension: " + std::to_string( EntityDimension )
      + ", region: " + std::string( TNL::convertToString( begin ).getString() ) + " - "
      + std::string( TNL::convertToString( end ).getString() ) );

   // expected results computed sequentially on the host
   HostGrid hostGrid;
   hostGrid.setSizes( grid.getSizes() );
   Real expectedSum = 0;
   Real expectedMax = std::numeric_limits< Real >::lowest();
   Index expectedCount = 0;
   TNL::Meshes::forEntities< EntityDimension >(
      hostGrid, begin, end, ExpectedResults< Real, Index >{ &expectedSum, &expectedMax, &expectedCount } );

   TNL::Containers::Vector< Real, Device, Index > values( grid.template getEntitiesCount< EntityDimension >() );
   values.forAllElements(
      [] __cuda_callable__( Index i, Real& value )
      {
         value = entityValue< Real >( i );
      } );
   const auto view = values.getConstView();
   auto fetch = [ = ] __cuda_callable__( const Entity& entity ) -> Real
   {
      return view[ entity.getIndex() ];
   };
   auto fetchOne = [] __cuda_callable__( const Entity& ) -> Index
   {
      return 1;
   };
   auto max = [] __cuda_callable__( const Real& a, const Real& b ) -> Real
   {
      return a > b ? a : b;
   };

   EXPECT_EQ( TNL::Meshes::reduce< EntityDimension >( grid, begin, end, fetch, TNL::Plus{}, (Real) 0 ), expectedSum );
   EXPECT_EQ( TNL::Meshes::reduce< EntityDimension >( grid, begin, end, fetch ), expectedSum );
   EXPECT_EQ( TNL::Meshes::reduce< EntityDimension >( grid, begin, end, fetch, TNL::Max{} ), expectedMax );
   EXPECT_EQ(
      TNL::Meshes::reduce< EntityDimension >( grid, begin, end, fetch, max, std::numeric_limits< Real >::lowest() ),
      expectedMax );
   EXPECT_EQ( TNL::Meshes::reduce< EntityDimension >( grid, begin, end, fetchOne ), expectedCount );
   if( wholeGrid ) {
      EXPECT_EQ( TNL::Meshes::reduce< EntityDimension >( grid, fetch, TNL::Plus{}, (Real) 0 ), expectedSum );
      EXPECT_EQ( TNL::Meshes::reduce< EntityDimension >( grid, fetch ), expectedSum );
      EXPECT_EQ( TNL::Meshes::reduce< EntityDimension >( grid, fetch, TNL::Max{} ), expectedMax );
      EXPECT_EQ(
         TNL::Meshes::reduce< EntityDimension >( grid, fetchOne ), grid.template getEntitiesCount< EntityDimension >() );
   }
}

template< typename Grid >
void
testReduce()
{
   constexpr int Dimension = Grid::getMeshDimension();
   using Coordinates = typename Grid::CoordinatesType;
   Grid grid;
   Coordinates sizes;
   for( int i = 0; i < Dimension; i++ )
      sizes[ i ] = 5 + 2 * i;
   grid.setSizes( sizes );

   // the whole grid, a general region, a region with thickness one and an empty region
   Coordinates regionBegin;
   Coordinates regionEnd;
   Coordinates thinEnd;
   for( int i = 0; i < Dimension; i++ ) {
      regionBegin[ i ] = 1 + i % 2;
      regionEnd[ i ] = sizes[ i ] - 1;
      thinEnd[ i ] = regionBegin[ i ] + ( i == 0 ? 1 : 2 );
   }
   const std::vector< std::pair< Coordinates, Coordinates > > regions = {
      { Coordinates( 0 ), sizes }, { regionBegin, regionEnd }, { regionBegin, thinEnd }, { regionBegin, regionBegin }
   };

   TNL::Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         for( std::size_t r = 0; r < regions.size(); r++ )
            testReduceRegion< decltype( entityDimension )::value >( grid, regions[ r ].first, regions[ r ].second, r == 0 );
      } );
}

template< typename Grid >
class GridReduceTest : public ::testing::Test
{};

template< int Dimension, typename Device >
using TestGrid = TNL::Meshes::Grid< Dimension, double, Device, int >;

using GridTypes = ::testing::Types<
#if defined( __CUDACC__ )
   TestGrid< 1, TNL::Devices::Cuda >,
   TestGrid< 2, TNL::Devices::Cuda >,
   TestGrid< 3, TNL::Devices::Cuda >,
   TNL::Meshes::Grid< 3, float, TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
   TestGrid< 1, TNL::Devices::Hip >,
   TestGrid< 2, TNL::Devices::Hip >,
   TestGrid< 3, TNL::Devices::Hip >,
   TNL::Meshes::Grid< 3, float, TNL::Devices::Hip, long >
#else
   TestGrid< 1, TNL::Devices::Sequential >,
   TestGrid< 2, TNL::Devices::Sequential >,
   TestGrid< 3, TNL::Devices::Sequential >,
   TestGrid< 1, TNL::Devices::Host >,
   TestGrid< 2, TNL::Devices::Host >,
   TestGrid< 3, TNL::Devices::Host >,
   TNL::Meshes::Grid< 3, float, TNL::Devices::Host, long >
#endif
   >;

TYPED_TEST_SUITE( GridReduceTest, GridTypes );

TYPED_TEST( GridReduceTest, reduce )
{
   testReduce< TypeParam >();
}

#include "../../main.h"
