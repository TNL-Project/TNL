#pragma once

#include <string>

#include <gtest/gtest.h>
#include <TNL/Algorithms/staticFor.h>
#include <TNL/Containers/Array.h>
#include <TNL/Meshes/GridEntitiesOrientations.h>

#include "../CoordinateIterator.h"

template< typename Device >
class GridAccessorsTestCaseInterface
{};

template< typename Device >
class GridAccessorsTestCase : public GridAccessorsTestCaseInterface< Device >
{};

template<>
class GridAccessorsTestCase< TNL::Devices::Sequential > : public GridAccessorsTestCaseInterface< TNL::Devices::Sequential >
{
public:
   template< typename Grid >
   void
   verifyDimensionGetters( const Grid& grid, const typename Grid::CoordinatesType& dimensions ) const
   {
      this->verifyDimensionByCoordinateGetter< Grid >( grid, dimensions );
   }

   template< typename Grid >
   void
   verifyEntitiesCountGetters(
      const Grid& grid,
      const TNL::Containers::StaticVector< Grid::getMeshDimension() + 1, typename Grid::IndexType >& entitiesCounts ) const
   {
      this->verifyEntitiesCountByContainerGetter< Grid >( grid, entitiesCounts );
      this->verifyEntitiesCountByIndexGetter< Grid >( grid, entitiesCounts );
   }

   template< typename Grid >
   void
   verifyOriginGetter( const Grid& grid, const typename Grid::PointType& coordinates ) const
   {
      auto result = grid.getOrigin();

      EXPECT_EQ( coordinates, result ) << "Verify, that the origin was correctly set";
   }

   template< typename Grid >
   void
   verifySpaceStepsGetter( const Grid& grid, const typename Grid::PointType& spaceSteps ) const
   {
      auto result = grid.getSpaceSteps();

      EXPECT_EQ( spaceSteps, result ) << "Verify, that space steps were correctly set";
   }

   template< typename Grid >
   void
   verifyDimensionByCoordinateGetter( const Grid& grid, const typename Grid::CoordinatesType& dimensions ) const
   {
      auto result = grid.getSizes();

      EXPECT_EQ( dimensions, result ) << "Verify, that dimension container accessor returns valid dimension";
   }

   template< typename Grid >
   void
   verifyEntitiesCountByContainerGetter(
      const Grid& grid,
      const TNL::Containers::StaticVector< Grid::getMeshDimension() + 1, typename Grid::IndexType >& entitiesCounts ) const
   {
      auto result = grid.getEntitiesCounts();

      EXPECT_EQ( entitiesCounts, result ) << "Verify, that returns expected entities counts";
   }

   template< typename Grid >
   void
   verifyEntitiesCountByIndexGetter(
      const Grid& grid,
      const TNL::Containers::StaticVector< Grid::getMeshDimension() + 1, typename Grid::IndexType >& entitiesCounts ) const
   {
      for( typename Grid::IndexType i = 0; i < entitiesCounts.getSize(); i++ )
         EXPECT_EQ( grid.getEntitiesCount( i ), entitiesCounts[ i ] ) << "Verify, that index access is correct.";
   }
};

template< typename Grid, typename... T >
void
testDimensionSetByCoordinate( Grid& grid, const typename Grid::CoordinatesType& dimensions )
{
   EXPECT_NO_THROW( grid.setSizes( dimensions ) ) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   SCOPED_TRACE( "Test dimension set by coordinate" );
   SCOPED_TRACE( "Dimension: " + TNL::convertToString( dimensions ) );
   SCOPED_TRACE( "Grid Dimension: " + TNL::convertToString( grid.getMeshDimension() ) );

   GridAccessorsTestCase< typename Grid::DeviceType > support;

   support.template verifyDimensionGetters< Grid >( grid, dimensions );
}

template< typename Grid >
void
testEntitiesCounts(
   Grid& grid,
   const typename Grid::CoordinatesType& dimensions,
   const TNL::Containers::StaticVector< Grid::getMeshDimension() + 1, typename Grid::IndexType >& entitiesCounts )
{
   EXPECT_NO_THROW( grid.setSizes( dimensions ) ) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   SCOPED_TRACE( "Test entities count" );
   SCOPED_TRACE( "Dimension: " + TNL::convertToString( dimensions ) );
   SCOPED_TRACE( "Grid Entities Counts: " + TNL::convertToString( grid.getEntitiesCounts() ) );
   SCOPED_TRACE( "Expected Entities Counts:: " + TNL::convertToString( entitiesCounts ) );

   GridAccessorsTestCase< typename Grid::DeviceType > support;

   support.template verifyEntitiesCountGetters< Grid >( grid, entitiesCounts );
}

template< typename Grid >
void
testOriginSetByCoordinate( Grid& grid, const typename Grid::PointType& coordinates )
{
   EXPECT_NO_THROW( grid.setOrigin( coordinates ) ) << "Verify, that the set of " << coordinates << " doesn't cause assert";

   SCOPED_TRACE( "Test origin set by index" );
   SCOPED_TRACE( "Coordinates: " + TNL::convertToString( coordinates ) );
   SCOPED_TRACE( "Grid origin: " + TNL::convertToString( grid.getOrigin() ) );

   GridAccessorsTestCase< typename Grid::DeviceType > support;

   support.template verifyOriginGetter< Grid >( grid, coordinates );
}

// Checks the space steps and proportions set by setDomain and setSpaceSteps, and that setSizes keeps
// the proportions of the domain and recomputes the space steps.
template< typename Grid >
void
testDomain( Grid& grid )
{
   using Coordinates = typename Grid::CoordinatesType;
   using Point = typename Grid::PointType;
   constexpr int Dimension = Grid::getMeshDimension();
   Coordinates sizes;
   Point origin;
   Point proportions;
   for( int i = 0; i < Dimension; i++ ) {
      sizes[ i ] = 2 + i;
      origin[ i ] = -1 - i;
      proportions[ i ] = 0.5 * ( i + 1 );
   }
   grid.setSizes( sizes );
   grid.setDomain( origin, proportions );
   EXPECT_EQ( grid.getOrigin(), origin );
   EXPECT_EQ( grid.getProportions(), proportions );
   for( int i = 0; i < Dimension; i++ )
      EXPECT_NEAR( grid.getSpaceSteps()[ i ], proportions[ i ] / sizes[ i ], 1e-6 ) << "Axis: " << i;

   // the proportions are kept when the sizes change
   grid.setSizes( 2 * sizes );
   EXPECT_EQ( grid.getOrigin(), origin );
   for( int i = 0; i < Dimension; i++ ) {
      EXPECT_NEAR( grid.getProportions()[ i ], proportions[ i ], 1e-6 ) << "Axis: " << i;
      EXPECT_NEAR( grid.getSpaceSteps()[ i ], proportions[ i ] / ( 2 * sizes[ i ] ), 1e-6 ) << "Axis: " << i;
   }

   // the space steps determine the proportions
   Point spaceSteps;
   for( int i = 0; i < Dimension; i++ )
      spaceSteps[ i ] = 0.25 * ( i + 1 );
   grid.setSpaceSteps( spaceSteps );
   EXPECT_EQ( grid.getSpaceSteps(), spaceSteps );
   for( int i = 0; i < Dimension; i++ )
      EXPECT_NEAR( grid.getProportions()[ i ], spaceSteps[ i ] * 2 * sizes[ i ], 1e-6 ) << "Axis: " << i;
}

// Checks that setSizes resets the local subdomain to the whole grid and all sides to the global boundary.
template< typename Grid >
void
testSetSizesResetsSubdomain( Grid& grid )
{
   using Coordinates = typename Grid::CoordinatesType;
   constexpr int Dimension = Grid::getMeshDimension();
   Coordinates sizes;
   for( int i = 0; i < Dimension; i++ )
      sizes[ i ] = 4 + i;
   grid.setSizes( sizes );
   EXPECT_EQ( grid.getLocalBegin(), Coordinates( 0 ) );
   EXPECT_EQ( grid.getLocalEnd(), sizes );
   EXPECT_EQ( grid.getLowerGlobalBoundarySides(), typename Grid::BoundarySidesType( true ) );
   EXPECT_EQ( grid.getUpperGlobalBoundarySides(), typename Grid::BoundarySidesType( true ) );

   grid.setLocalSubdomain( Coordinates( 1 ), sizes - 1 );
   grid.setGlobalBoundarySides( typename Grid::BoundarySidesType( false ), typename Grid::BoundarySidesType( false ) );
   EXPECT_EQ( grid.getLocalBegin(), Coordinates( 1 ) );
   EXPECT_EQ( grid.getLocalEnd(), sizes - 1 );

   grid.setSizes( sizes + 1 );
   EXPECT_EQ( grid.getLocalBegin(), Coordinates( 0 ) );
   EXPECT_EQ( grid.getLocalEnd(), sizes + 1 );
   EXPECT_EQ( grid.getLowerGlobalBoundarySides(), typename Grid::BoundarySidesType( true ) );
   EXPECT_EQ( grid.getUpperGlobalBoundarySides(), typename Grid::BoundarySidesType( true ) );
}

// Checks that a grid with zero sizes has no entities and the traversals do not visit any entity.
template< typename Grid >
void
testEmptyGrid( Grid& grid )
{
   constexpr int Dimension = Grid::getMeshDimension();
   grid.setSizes( typename Grid::CoordinatesType( 0 ) );
   TNL::Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         constexpr int EntityDimension = decltype( entityDimension )::value;
         using Entity = typename Grid::template EntityType< EntityDimension >;
         EXPECT_EQ( grid.template getEntitiesCount< EntityDimension >(), 0 ) << "Entity dimension: " << EntityDimension;
         int visits = 0;
         auto count = [ & ]( const Entity& )
         {
            visits++;
         };
         TNL::Meshes::forAllEntities< EntityDimension >( grid, count );
         TNL::Meshes::forInteriorEntities< EntityDimension >( grid, count );
         TNL::Meshes::forBoundaryEntities< EntityDimension >( grid, count );
         TNL::Meshes::forLocalEntities< EntityDimension >( grid, count );
         EXPECT_EQ( visits, 0 ) << "Entity dimension: " << EntityDimension;
      } );
}

// Checks that a grid entity created without explicit orientation gets the first orientation of its dimension.
template< typename Grid, int EntityDimension >
void
testDefaultEntityOrientation( Grid& grid )
{
   using Entity = typename Grid::template EntityType< EntityDimension >;
   using Coordinates = typename Grid::CoordinatesType;
   using Index = typename Grid::IndexType;
   using EntitiesOrientations = TNL::Meshes::GridEntitiesOrientations< Grid::getMeshDimension() >;

   SCOPED_TRACE( "Entity dimension: " + std::to_string( EntityDimension ) );

   grid.setSizes( Coordinates( 3 ) );
   const Coordinates coordinates( 1 );
   const Index expectedTotalOrientation = EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >( 0 );
   const auto expectedNormals = EntitiesOrientations::template getNormals< EntityDimension, 0 >();

   const Entity reference( grid, coordinates, Index( 0 ) );
   const Entity entity( grid, coordinates );
   EXPECT_EQ( entity.getTotalOrientationIndex(), expectedTotalOrientation );
   EXPECT_EQ( entity.getOrientationIndex(), 0 );
   EXPECT_EQ( entity.getNormals(), expectedNormals );
   EXPECT_EQ( entity.getIndex(), reference.getIndex() );

   const Entity defaultEntity;
   EXPECT_EQ( defaultEntity.getTotalOrientationIndex(), expectedTotalOrientation );

   const Entity entityFromCoordinates( coordinates );
   EXPECT_EQ( entityFromCoordinates.getTotalOrientationIndex(), expectedTotalOrientation );
}

// Checks that the basis of entities with given dimension and orientation is the complement of their normals.
template< typename Grid >
void
testBasis( const Grid& grid )
{
   using EntitiesOrientations = TNL::Meshes::GridEntitiesOrientations< Grid::getMeshDimension() >;
   TNL::Algorithms::staticFor< int, 0, Grid::getMeshDimension() + 1 >(
      [ & ]( auto entityDimension )
      {
         constexpr int EntityDimension = decltype( entityDimension )::value;
         EntitiesOrientations entitiesOrientations;
         for( int orientation = 0; orientation < grid.getEntityOrientationsCount( EntityDimension ); orientation++ ) {
            const typename Grid::CoordinatesType normals =
               entitiesOrientations.template getNormals< EntityDimension >( orientation );
            EXPECT_EQ( grid.template getBasis< EntityDimension >( orientation ), 1 - normals )
               << "Entity dimension: " << EntityDimension << ", orientation: " << orientation;
         }
      } );
}
