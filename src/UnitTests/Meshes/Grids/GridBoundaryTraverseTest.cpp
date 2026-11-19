#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/Grid.h>

// Checks whether the entity lies on the boundary of the region [ begin, end ). The entities with given orientation
// lie in the box [ begin, end + normals ) and they are on its boundary if their coordinate along some boundary
// axis is the first or the last one in the box. The boundary axes are all axes for cells and the axes with
// nonzero normals for the other entities.
template< typename Entity, typename Coordinates >
bool
isOnRegionBoundary( const Entity& entity, const Coordinates& begin, const Coordinates& end )
{
   constexpr int Dimension = Entity::getMeshDimension();
   const auto normals = entity.getNormals();
   const auto& coordinates = entity.getCoordinates();
   bool inside = true;
   bool boundary = false;
   for( int i = 0; i < Dimension; i++ ) {
      const int regionEnd = end[ i ] + normals[ i ];
      if( coordinates[ i ] < begin[ i ] || coordinates[ i ] >= regionEnd )
         inside = false;
      const bool boundaryAxis = Entity::getEntityDimension() == Dimension || normals[ i ];
      if( boundaryAxis && ( coordinates[ i ] == begin[ i ] || coordinates[ i ] == regionEnd - 1 ) )
         boundary = true;
   }
   return inside && boundary;
}

// Checks that forBoundaryEntities visits each entity on the boundary of the region exactly once and no other entity.
template< typename Grid, int EntityDimension >
void
testRegionBoundary( const Grid& grid, const typename Grid::CoordinatesType& begin, const typename Grid::CoordinatesType& end )
{
   using Entity = typename Grid::template EntityType< EntityDimension >;
   SCOPED_TRACE(
      "Entity dimension: " + std::to_string( EntityDimension )
      + ", region: " + std::string( TNL::convertToString( begin ).getString() ) + " - "
      + std::string( TNL::convertToString( end ).getString() ) );

   std::vector< int > calls( grid.template getEntitiesCount< EntityDimension >(), 0 );
   grid.template forBoundaryEntities< EntityDimension >(
      begin,
      end,
      [ & ]( const Entity& entity )
      {
         calls.at( entity.getIndex() )++;
      } );

   grid.template forAllEntities< EntityDimension >(
      [ & ]( const Entity& entity )
      {
         const int expectedCalls = isOnRegionBoundary( entity, begin, end ) ? 1 : 0;
         EXPECT_EQ( calls[ entity.getIndex() ], expectedCalls ) << "Entity coordinates: " << entity.getCoordinates()
            << ", normals: " << entity.getNormals();
      } );
}

template< int Dimension >
void
testRegionBoundaryForAllEntities()
{
   using Grid = TNL::Meshes::Grid< Dimension, double, TNL::Devices::Sequential, int >;
   using Coordinates = typename Grid::CoordinatesType;
   Grid grid;
   Coordinates sizes;
   for( int i = 0; i < Dimension; i++ )
      sizes[ i ] = 5 + i;
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
         for( const auto& region : regions )
            testRegionBoundary< Grid, decltype( entityDimension )::value >( grid, region.first, region.second );
      } );
}

TEST( GridBoundaryTraverseTest, RegionBoundary_1D )
{
   testRegionBoundaryForAllEntities< 1 >();
}

TEST( GridBoundaryTraverseTest, RegionBoundary_2D )
{
   testRegionBoundaryForAllEntities< 2 >();
}

TEST( GridBoundaryTraverseTest, RegionBoundary_3D )
{
   testRegionBoundaryForAllEntities< 3 >();
}

// Checks that the traversals with a launch configuration visit the same entities as the traversals without it.
template< int Dimension >
void
testLaunchConfiguration()
{
   using Grid = TNL::Meshes::Grid< Dimension, double, TNL::Devices::Sequential, int >;
   using Coordinates = typename Grid::CoordinatesType;
   using LaunchConfiguration = typename Grid::DeviceType::LaunchConfiguration;
   Grid grid;
   Coordinates sizes;
   Coordinates regionBegin;
   for( int i = 0; i < Dimension; i++ ) {
      sizes[ i ] = 4 + i;
      regionBegin[ i ] = 1;
   }
   grid.setSizes( sizes );

   TNL::Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         constexpr int EntityDimension = decltype( entityDimension )::value;
         using Entity = typename Grid::template EntityType< EntityDimension >;
         const int count = grid.template getEntitiesCount< EntityDimension >();
         std::vector< int > expected( count, 0 );
         std::vector< int > calls( count, 0 );
         auto expect = [ & ]( const Entity& entity )
         {
            expected.at( entity.getIndex() )++;
         };
         auto call = [ & ]( const Entity& entity )
         {
            calls.at( entity.getIndex() )++;
         };
         const LaunchConfiguration launchConfiguration;

         grid.template forAllEntities< EntityDimension >( expect );
         grid.template forAllEntities< EntityDimension >( launchConfiguration, call );
         grid.template forInteriorEntities< EntityDimension >( expect );
         grid.template forInteriorEntities< EntityDimension >( launchConfiguration, call );
         grid.template forBoundaryEntities< EntityDimension >( expect );
         grid.template forBoundaryEntities< EntityDimension >( launchConfiguration, call );
         grid.template forEntities< EntityDimension >( regionBegin, sizes, expect );
         grid.template forEntities< EntityDimension >( regionBegin, sizes, launchConfiguration, call );
         grid.template forBoundaryEntities< EntityDimension >( regionBegin, sizes, expect );
         grid.template forBoundaryEntities< EntityDimension >( regionBegin, sizes, launchConfiguration, call );
         EXPECT_EQ( calls, expected ) << "Entity dimension: " << EntityDimension;
      } );
}

TEST( GridBoundaryTraverseTest, LaunchConfiguration_1D )
{
   testLaunchConfiguration< 1 >();
}

TEST( GridBoundaryTraverseTest, LaunchConfiguration_2D )
{
   testLaunchConfiguration< 2 >();
}

TEST( GridBoundaryTraverseTest, LaunchConfiguration_3D )
{
   testLaunchConfiguration< 3 >();
}

#include "../../main.h"
