#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/staticFor.h>
#include <TNL/Containers/Array.h>
#include <TNL/Functional.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/reduce.h>

// The traversals and reductions on all devices are compared with the sequential ones, which are checked against
// independent references in GridBoundaryTraverseTest. The grids include a local subdomain with only some sides on
// the global boundary, as in distributed grids, and the traversals on GPUs are launched without blocking the host.

namespace GridTraverseDeviceTest {

using namespace TNL;
using namespace TNL::Meshes;

// Counts the visits of the entities. Atomic operations are used so that multiple visits are not lost.
template< typename View >
struct CountVisits
{
   mutable View view;

   template< typename Entity >
   __cuda_callable__
   void
   operator()( const Entity& entity ) const
   {
      Algorithms::AtomicOperations< typename View::DeviceType >::add( view[ entity.getIndex() ], 1 );
   }
};

// Value of an entity for the reductions, different for different entities.
struct EntityValue
{
   template< typename Entity >
   __cuda_callable__
   double
   operator()( const Entity& entity ) const
   {
      return entity.getIndex() + 1;
   }
};

enum class Traversal
{
   All,
   Interior,
   Boundary,
   Local,
   Region,
   RegionBoundary
};

constexpr Traversal traversals[] = { Traversal::All,   Traversal::Interior, Traversal::Boundary,
                                     Traversal::Local, Traversal::Region,   Traversal::RegionBoundary };

std::string
getTraversalName( Traversal traversal )
{
   switch( traversal ) {
      case Traversal::All:
         return "all";
      case Traversal::Interior:
         return "interior";
      case Traversal::Boundary:
         return "boundary";
      case Traversal::Local:
         return "local";
      case Traversal::Region:
         return "region";
      case Traversal::RegionBoundary:
         return "region boundary";
   }
   return "";
}

// Sets up the grid: variant 0 is a grid which is not distributed, variant 1 has a local subdomain with the lower
// global boundary only along the odd axes and the upper global boundary only along the even axes.
template< typename Grid >
void
setUpGrid( Grid& grid, int variant )
{
   constexpr int Dimension = Grid::getMeshDimension();
   typename Grid::CoordinatesType sizes;
   for( int i = 0; i < Dimension; i++ )
      sizes[ i ] = 5 + i;
   grid.setSizes( sizes );
   if( variant == 1 ) {
      typename Grid::CoordinatesType begin;
      typename Grid::CoordinatesType end;
      typename Grid::BoundarySidesType lower;
      typename Grid::BoundarySidesType upper;
      for( int i = 0; i < Dimension; i++ ) {
         lower[ i ] = i % 2 == 1;
         upper[ i ] = i % 2 == 0;
         begin[ i ] = lower[ i ] ? 0 : 1;
         end[ i ] = upper[ i ] ? sizes[ i ] : sizes[ i ] - 1;
      }
      grid.setLocalSubdomain( begin, end );
      grid.setGlobalBoundarySides( lower, upper );
   }
}

// The region used by the region traversals.
template< typename Grid >
std::pair< typename Grid::CoordinatesType, typename Grid::CoordinatesType >
getRegion( const Grid& grid )
{
   typename Grid::CoordinatesType begin;
   typename Grid::CoordinatesType end;
   for( int i = 0; i < Grid::getMeshDimension(); i++ ) {
      begin[ i ] = 1;
      end[ i ] = grid.getSizes()[ i ] - i % 2;
   }
   return { begin, end };
}

template< int EntityDimension, typename Grid, typename Func >
void
traverse( const Grid& grid, Traversal traversal, Func func, typename Grid::DeviceType::LaunchConfiguration launchConfiguration )
{
   const auto region = getRegion( grid );
   switch( traversal ) {
      case Traversal::All:
         forAllEntities< EntityDimension >( grid, func, launchConfiguration );
         break;
      case Traversal::Interior:
         forInteriorEntities< EntityDimension >( grid, func, launchConfiguration );
         break;
      case Traversal::Boundary:
         forBoundaryEntities< EntityDimension >( grid, func, launchConfiguration );
         break;
      case Traversal::Local:
         forLocalEntities< EntityDimension >( grid, func, launchConfiguration );
         break;
      case Traversal::Region:
         forEntities< EntityDimension >( grid, region.first, region.second, func, launchConfiguration );
         break;
      case Traversal::RegionBoundary:
         forBoundaryEntities< EntityDimension >( grid, region.first, region.second, func, launchConfiguration );
         break;
   }
}

template< typename Device >
class GridTraverseDeviceTest : public ::testing::Test
{
protected:
   using DeviceType = Device;
};

using Devices = ::testing::Types<
#if defined( __CUDACC__ ) || defined( __HIP__ )
   TNL::Devices::GPU
#else
   TNL::Devices::Host
#endif
   >;

TYPED_TEST_SUITE( GridTraverseDeviceTest, Devices );

template< typename Device, int Dimension, int EntityDimension >
void
testTraversals()
{
   using DeviceGrid = Grid< Dimension, double, Device, int >;
   using SequentialGrid = Grid< Dimension, double, TNL::Devices::Sequential, int >;
   using DeviceArray = Containers::Array< int, Device, int >;
   using SequentialArray = Containers::Array< int, TNL::Devices::Sequential, int >;

   for( int variant = 0; variant < 2; variant++ ) {
      DeviceGrid grid;
      SequentialGrid sequentialGrid;
      setUpGrid( grid, variant );
      setUpGrid( sequentialGrid, variant );
      const int count = grid.template getEntitiesCount< EntityDimension >();

      for( Traversal traversal : traversals ) {
         SCOPED_TRACE(
            "Dimension: " + std::to_string( Dimension ) + ", entity dimension: " + std::to_string( EntityDimension )
            + ", variant: " + std::to_string( variant ) + ", traversal: " + getTraversalName( traversal ) );

         typename Device::LaunchConfiguration launchConfiguration;
         if constexpr( std::is_same_v< Device, TNL::Devices::GPU > ) {
            launchConfiguration.blockSize.x = 64;
            launchConfiguration.blockHostUntilFinished = false;
         }
         DeviceArray visits( count, 0 );
         traverse< EntityDimension >(
            grid, traversal, CountVisits< typename DeviceArray::ViewType >{ visits.getView() }, launchConfiguration );
         if constexpr( std::is_same_v< Device, TNL::Devices::GPU > )
            Backend::deviceSynchronize();

         SequentialArray expected( count, 0 );
         traverse< EntityDimension >(
            sequentialGrid, traversal, CountVisits< typename SequentialArray::ViewType >{ expected.getView() }, {} );
         EXPECT_EQ( SequentialArray( visits ), expected );
      }

      SCOPED_TRACE(
         "Dimension: " + std::to_string( Dimension ) + ", entity dimension: " + std::to_string( EntityDimension )
         + ", variant: " + std::to_string( variant ) );
      EXPECT_EQ(
         reduceLocalEntities< EntityDimension >( grid, EntityValue{} ),
         reduceLocalEntities< EntityDimension >( sequentialGrid, EntityValue{} ) );
      EXPECT_EQ( reduce< EntityDimension >( grid, EntityValue{} ), reduce< EntityDimension >( sequentialGrid, EntityValue{} ) );
      const auto region = getRegion( grid );
      EXPECT_EQ(
         reduce< EntityDimension >( grid, region.first, region.second, EntityValue{} ),
         reduce< EntityDimension >( sequentialGrid, region.first, region.second, EntityValue{} ) );
   }
}

template< typename Device, int Dimension >
void
testTraversalsForAllEntities()
{
   Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         testTraversals< Device, Dimension, decltype( entityDimension )::value >();
      } );
}

TYPED_TEST( GridTraverseDeviceTest, Grid_1D )
{
   testTraversalsForAllEntities< typename TestFixture::DeviceType, 1 >();
}

TYPED_TEST( GridTraverseDeviceTest, Grid_2D )
{
   testTraversalsForAllEntities< typename TestFixture::DeviceType, 2 >();
}

TYPED_TEST( GridTraverseDeviceTest, Grid_3D )
{
   testTraversalsForAllEntities< typename TestFixture::DeviceType, 3 >();
}

}  // namespace GridTraverseDeviceTest

#include "../../main.h"
