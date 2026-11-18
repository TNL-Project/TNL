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

template< typename... Parameters >
std::string
makeString( Parameters... parameters )
{
   std::ostringstream s;

   for( const auto x : { parameters... } )
      s << x << ", ";

   return s.str();
}

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

template< typename Grid >
void
testSpaceStepsSetByCoordinate( Grid& grid, const int spaceStepsSize, const typename Grid::PointType& spaceSteps )
{
   EXPECT_NO_THROW( grid.setSpaceSteps( spaceSteps ) ) << "Verify, that the set of " << spaceSteps << " doesn't cause assert";

   SCOPED_TRACE( "Test space steps set by coordinate" );
   SCOPED_TRACE( "Space steps: " + TNL::convertToString( spaceSteps ) );
   SCOPED_TRACE( "Grid space steps: " + TNL::convertToString( grid.getSpaceSteps() ) );

   GridAccessorsTestCase< typename Grid::DeviceType > support;

   support.template verifySpaceStepsGetter< Grid >( grid, spaceSteps );
   support.template verifySpaceStepsValues< Grid >( grid, spaceStepsSize, spaceSteps );
}

template<
   typename Grid,
   typename... T,
   std::enable_if_t< TNL::Meshes::Templates::conjunction_v< std::is_convertible< typename Grid::RealType, T >... >, bool > =
      true >
void
testSpaceStepsSetByIndex( Grid& grid, const int spaceStepsSize, T... spaceSteps )
{
   typename Grid::PointType spaceStepsContainer( spaceSteps... );

   EXPECT_NO_THROW( grid.setSpaceSteps( spaceSteps... ) )
      << "Verify, that the set of " << spaceStepsContainer << " doesn't cause assert";

   SCOPED_TRACE( "Test space steps set by index" );
   SCOPED_TRACE( "Space steps: " + TNL::convertToString( spaceStepsContainer ) );
   SCOPED_TRACE( "Grid space steps: " + TNL::convertToString( grid.getSpaceSteps() ) );

   GridAccessorsTestCase< typename Grid::DeviceType > support;

   support.template verifySpaceStepsGetter< Grid >( grid, spaceStepsContainer );
   support.template verifySpaceStepsValues< Grid >( grid, spaceStepsSize, spaceStepsContainer );
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
