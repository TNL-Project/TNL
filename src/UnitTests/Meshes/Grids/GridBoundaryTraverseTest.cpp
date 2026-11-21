#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Functional.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/reduce.h>

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
   TNL::Meshes::forBoundaryEntities< EntityDimension >(
      grid,
      begin,
      end,
      [ & ]( const Entity& entity )
      {
         calls.at( entity.getIndex() )++;
      } );

   TNL::Meshes::forAllEntities< EntityDimension >(
      grid,
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

         TNL::Meshes::forAllEntities< EntityDimension >( grid, expect );
         TNL::Meshes::forAllEntities< EntityDimension >( grid, call, launchConfiguration );
         TNL::Meshes::forInteriorEntities< EntityDimension >( grid, expect );
         TNL::Meshes::forInteriorEntities< EntityDimension >( grid, call, launchConfiguration );
         TNL::Meshes::forBoundaryEntities< EntityDimension >( grid, expect );
         TNL::Meshes::forBoundaryEntities< EntityDimension >( grid, call, launchConfiguration );
         TNL::Meshes::forEntities< EntityDimension >( grid, regionBegin, sizes, expect );
         TNL::Meshes::forEntities< EntityDimension >( grid, regionBegin, sizes, call, launchConfiguration );
         TNL::Meshes::forBoundaryEntities< EntityDimension >( grid, regionBegin, sizes, expect );
         TNL::Meshes::forBoundaryEntities< EntityDimension >( grid, regionBegin, sizes, call, launchConfiguration );
         EXPECT_EQ( calls, expected ) << "Entity dimension: " << EntityDimension;
      } );
}

// Checks that isBoundaryEntity agrees with forBoundaryEntities and forInteriorEntities for a grid with a local
// subdomain which does not start at the origin of the grid and all sides on the global boundary.
template< int Dimension >
void
testBoundaryOfLocalSubdomain()
{
   using Grid = TNL::Meshes::Grid< Dimension, double, TNL::Devices::Sequential, int >;
   using Coordinates = typename Grid::CoordinatesType;
   Grid grid;
   Coordinates sizes;
   for( int i = 0; i < Dimension; i++ )
      sizes[ i ] = 4 + i;
   grid.setSizes( sizes );
   grid.setLocalSubdomain( Coordinates( 1 ), sizes - 1 );

   TNL::Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         constexpr int EntityDimension = decltype( entityDimension )::value;
         using Entity = typename Grid::template EntityType< EntityDimension >;
         SCOPED_TRACE( "Entity dimension: " + std::to_string( EntityDimension ) );
         std::vector< int > boundaryCalls( grid.template getEntitiesCount< EntityDimension >(), 0 );
         std::vector< int > interiorCalls( grid.template getEntitiesCount< EntityDimension >(), 0 );
         TNL::Meshes::forBoundaryEntities< EntityDimension >(
            grid,
            [ & ]( const Entity& entity )
            {
               boundaryCalls.at( entity.getIndex() )++;
            } );
         TNL::Meshes::forInteriorEntities< EntityDimension >(
            grid,
            [ & ]( const Entity& entity )
            {
               interiorCalls.at( entity.getIndex() )++;
            } );
         TNL::Meshes::forLocalEntities< EntityDimension >(
            grid,
            [ & ]( const Entity& entity )
            {
               const bool boundary = grid.isBoundaryEntity( entity );
               EXPECT_EQ( boundaryCalls[ entity.getIndex() ], boundary ? 1 : 0 )
                  << "Entity coordinates: " << entity.getCoordinates() << ", normals: " << entity.getNormals();
               EXPECT_EQ( interiorCalls[ entity.getIndex() ], boundary ? 0 : 1 )
                  << "Entity coordinates: " << entity.getCoordinates() << ", normals: " << entity.getNormals();
            } );
      } );
}

TEST( GridBoundaryTraverseTest, BoundaryOfLocalSubdomain_1D )
{
   testBoundaryOfLocalSubdomain< 1 >();
}

TEST( GridBoundaryTraverseTest, BoundaryOfLocalSubdomain_2D )
{
   testBoundaryOfLocalSubdomain< 2 >();
}

TEST( GridBoundaryTraverseTest, BoundaryOfLocalSubdomain_3D )
{
   testBoundaryOfLocalSubdomain< 3 >();
}

// Emulates the decomposition of a global grid into subdomains in the same way as DistributedGrid: the local grid
// of each subdomain has one layer of ghost cells on the sides adjoining other subdomains and only the other sides
// lie on the global boundary. Checks that
// - the owned entities of all subdomains are exactly the entities of the global grid, each one owned once,
// - the boundary and interior entities of all subdomains are exactly the boundary and interior entities of the
//   global grid, each one visited once,
// - isBoundaryEntity agrees with forBoundaryEntities and reduceLocalEntities with forLocalEntities.
template< int Dimension, int EntityDimension >
void
testDecomposition(
   const typename TNL::Meshes::Grid< Dimension, double, TNL::Devices::Sequential, int >::CoordinatesType& decomposition )
{
   using Grid = TNL::Meshes::Grid< Dimension, double, TNL::Devices::Sequential, int >;
   using Coordinates = typename Grid::CoordinatesType;
   using Entity = typename Grid::template EntityType< EntityDimension >;
   SCOPED_TRACE(
      "Entity dimension: " + std::to_string( EntityDimension )
      + ", decomposition: " + std::string( TNL::convertToString( decomposition ).getString() ) );

   Coordinates globalSizes;
   for( int i = 0; i < Dimension; i++ )
      globalSizes[ i ] = 5 + i;
   Grid globalGrid;
   globalGrid.setSizes( globalSizes );
   const int globalCount = globalGrid.template getEntitiesCount< EntityDimension >();
   std::vector< int > localCalls( globalCount, 0 );
   std::vector< int > boundaryCalls( globalCount, 0 );
   std::vector< int > interiorCalls( globalCount, 0 );

   int subdomainsCount = 1;
   for( int i = 0; i < Dimension; i++ )
      subdomainsCount *= decomposition[ i ];
   for( int s = 0; s < subdomainsCount; s++ ) {
      // the subdomain coordinates, the sizes and the overlaps as in DistributedGrid
      Coordinates globalBegin;
      Coordinates localSize;
      Coordinates lower;
      Coordinates upper;
      typename Grid::BoundarySidesType lowerSides;
      typename Grid::BoundarySidesType upperSides;
      int tmp = s;
      for( int i = 0; i < Dimension; i++ ) {
         const int subdomain = tmp % decomposition[ i ];
         tmp /= decomposition[ i ];
         const int size = globalSizes[ i ] / decomposition[ i ];
         const int larger = globalSizes[ i ] % decomposition[ i ];
         localSize[ i ] = size + ( subdomain < larger ? 1 : 0 );
         globalBegin[ i ] = subdomain * size + std::min( subdomain, larger );
         lowerSides[ i ] = subdomain == 0;
         upperSides[ i ] = subdomain == decomposition[ i ] - 1;
         lower[ i ] = lowerSides[ i ] ? 0 : 1;
         upper[ i ] = upperSides[ i ] ? 0 : 1;
      }
      Grid localGrid;
      localGrid.setSizes( localSize + lower + upper );
      localGrid.setLocalSubdomain( lower, localGrid.getSizes() - upper );
      localGrid.setGlobalBoundarySides( lowerSides, upperSides );

      auto getGlobalIndex = [ & ]( const Entity& entity )
      {
         const Entity globalEntity( globalGrid, entity.getCoordinates() - lower + globalBegin, entity.getNormals() );
         return globalEntity.getIndex();
      };

      std::vector< int > localBoundaryCalls( localGrid.template getEntitiesCount< EntityDimension >(), 0 );
      int localCount = 0;
      TNL::Meshes::forLocalEntities< EntityDimension >(
         localGrid,
         [ & ]( const Entity& entity )
         {
            localCalls.at( getGlobalIndex( entity ) )++;
            localCount++;
         } );
      TNL::Meshes::forBoundaryEntities< EntityDimension >(
         localGrid,
         [ & ]( const Entity& entity )
         {
            boundaryCalls.at( getGlobalIndex( entity ) )++;
            localBoundaryCalls.at( entity.getIndex() )++;
         } );
      TNL::Meshes::forInteriorEntities< EntityDimension >(
         localGrid,
         [ & ]( const Entity& entity )
         {
            interiorCalls.at( getGlobalIndex( entity ) )++;
         } );
      TNL::Meshes::forLocalEntities< EntityDimension >(
         localGrid,
         [ & ]( const Entity& entity )
         {
            EXPECT_EQ( localGrid.isBoundaryEntity( entity ), localBoundaryCalls[ entity.getIndex() ] == 1 ) << "Subdomain: " << s
               << ", entity coordinates: " << entity.getCoordinates() << ", normals: " << entity.getNormals();
         } );
      const int reducedCount = TNL::Meshes::reduceLocalEntities< EntityDimension >(
         localGrid,
         []( const Entity& entity )
         {
            return 1;
         },
         TNL::Plus{},
         0 );
      EXPECT_EQ( reducedCount, localCount ) << "Subdomain: " << s;
   }

   TNL::Meshes::forAllEntities< EntityDimension >(
      globalGrid,
      [ & ]( const Entity& entity )
      {
         const int index = entity.getIndex();
         const bool boundary = globalGrid.isBoundaryEntity( entity );
         EXPECT_EQ( localCalls[ index ], 1 ) << "Entity coordinates: " << entity.getCoordinates()
            << ", normals: " << entity.getNormals();
         EXPECT_EQ( boundaryCalls[ index ], boundary ? 1 : 0 ) << "Entity coordinates: " << entity.getCoordinates()
            << ", normals: " << entity.getNormals();
         EXPECT_EQ( interiorCalls[ index ], boundary ? 0 : 1 ) << "Entity coordinates: " << entity.getCoordinates()
            << ", normals: " << entity.getNormals();
      } );
}

template< int Dimension >
void
testDecompositions()
{
   using Coordinates = typename TNL::Meshes::Grid< Dimension, double, TNL::Devices::Sequential, int >::CoordinatesType;
   // all decompositions with 1 to 3 subdomains along each axis
   int decompositionsCount = 1;
   for( int i = 0; i < Dimension; i++ )
      decompositionsCount *= 3;
   for( int d = 0; d < decompositionsCount; d++ ) {
      Coordinates decomposition;
      int tmp = d;
      for( int i = 0; i < Dimension; i++ ) {
         decomposition[ i ] = tmp % 3 + 1;
         tmp /= 3;
      }
      TNL::Algorithms::staticFor< int, 0, Dimension + 1 >(
         [ & ]( auto entityDimension )
         {
            testDecomposition< Dimension, decltype( entityDimension )::value >( decomposition );
         } );
   }
}

TEST( GridBoundaryTraverseTest, Decomposition_1D )
{
   testDecompositions< 1 >();
}

TEST( GridBoundaryTraverseTest, Decomposition_2D )
{
   testDecompositions< 2 >();
}

TEST( GridBoundaryTraverseTest, Decomposition_3D )
{
   testDecompositions< 3 >();
}

// Checks that the deprecated member functions of the grid, which pass additional arguments to the function,
// visit the same entities as the free functions.
template< int Dimension >
void
testDeprecatedMemberFunctions()
{
   using Grid = TNL::Meshes::Grid< Dimension, double, TNL::Devices::Sequential, int >;
   using Coordinates = typename Grid::CoordinatesType;
   Grid grid;
   Coordinates sizes;
   Coordinates regionBegin;
   for( int i = 0; i < Dimension; i++ ) {
      sizes[ i ] = 4 + i;
      regionBegin[ i ] = 1;
   }
   grid.setSizes( sizes );
   grid.setLocalSubdomain( regionBegin, sizes );

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
         auto call = [ & ]( const Entity& entity, int increment )
         {
            calls.at( entity.getIndex() ) += increment;
         };

         TNL::Meshes::forAllEntities< EntityDimension >( grid, expect );
         TNL::Meshes::forInteriorEntities< EntityDimension >( grid, expect );
         TNL::Meshes::forBoundaryEntities< EntityDimension >( grid, expect );
         TNL::Meshes::forEntities< EntityDimension >( grid, regionBegin, sizes, expect );
         TNL::Meshes::forBoundaryEntities< EntityDimension >( grid, regionBegin, sizes, expect );
         TNL::Meshes::forLocalEntities< EntityDimension >( grid, expect );
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
         grid.template forAllEntities< EntityDimension >( call, 1 );
         grid.template forInteriorEntities< EntityDimension >( call, 1 );
         grid.template forBoundaryEntities< EntityDimension >( call, 1 );
         grid.template forEntities< EntityDimension >( regionBegin, sizes, call, 1 );
         grid.template forBoundaryEntities< EntityDimension >( regionBegin, sizes, call, 1 );
         grid.template forLocalEntities< EntityDimension >( call, 1 );
#pragma GCC diagnostic pop
         EXPECT_EQ( calls, expected ) << "Entity dimension: " << EntityDimension;
      } );
}

TEST( GridBoundaryTraverseTest, DeprecatedMemberFunctions_1D )
{
   testDeprecatedMemberFunctions< 1 >();
}

TEST( GridBoundaryTraverseTest, DeprecatedMemberFunctions_2D )
{
   testDeprecatedMemberFunctions< 2 >();
}

TEST( GridBoundaryTraverseTest, DeprecatedMemberFunctions_3D )
{
   testDeprecatedMemberFunctions< 3 >();
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
