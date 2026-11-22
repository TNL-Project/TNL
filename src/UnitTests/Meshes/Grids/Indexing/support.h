#pragma once

#include <vector>

#include "../ReferenceEntityIndex.h"

// Checks that the entity constructed from the index of another entity has the same coordinates and orientation,
// that the index agrees with the index computed independently by getReferenceEntityIndex and that the indexes of
// all entities form a permutation of [ 0, count ).
template< typename Grid, int EntityDimension >
void
testGetEntityFromIndex(
   Grid& grid,
   const typename Grid::CoordinatesType& dimensions,
   const typename Grid::PointType& origin = typename Grid::PointType( 0 ),
   const typename Grid::PointType& spaceSteps = typename Grid::PointType( 1 ) )
{
   SCOPED_TRACE( "Grid dimension: " + TNL::convertToString( Grid::getMeshDimension() ) );
   SCOPED_TRACE( "Grid dimensions: " + TNL::convertToString( dimensions ) );

   EXPECT_NO_THROW( grid.setSizes( dimensions ) ) << "Verify, that the set of" << dimensions << " doesn't cause assert";
   EXPECT_NO_THROW( grid.setOrigin( origin ) ) << "Verify, that the set of" << origin << "doesn't cause assert";
   EXPECT_NO_THROW( grid.setSpaceSteps( spaceSteps ) ) << "Verify, that the set of" << spaceSteps << "doesn't cause assert";

   SCOPED_TRACE( "Entity dimension: " + TNL::convertToString( EntityDimension ) );

   const int count = grid.template getEntitiesCount< EntityDimension >();
   std::vector< int > calls( count, 0 );
   TNL::Meshes::forAllEntities< EntityDimension >(
      grid,
      [ & ]( TNL::Meshes::GridEntity< Grid, EntityDimension >& entity ) mutable
      {
         auto new_entity = TNL::Meshes::GridEntity< Grid, EntityDimension >( entity.getGrid(), entity.getIndex() );
         EXPECT_EQ( new_entity.getCoordinates(), entity.getCoordinates() ) << " Entity coordinates: " << entity.getCoordinates()
            << std::endl
            << " New entity coordinates: " << new_entity.getCoordinates() << std::endl;
         EXPECT_EQ( new_entity.getNormals(), entity.getNormals() ) << " Entity normals: " << entity.getNormals() << std::endl
            << " New entity normals: " << new_entity.getNormals();
         EXPECT_EQ( new_entity.getOrientationIndex(), entity.getOrientationIndex() );
         EXPECT_EQ( entity.getIndex(), getReferenceEntityIndex( grid, entity.getCoordinates(), entity.getNormals() ) )
            << " Entity coordinates: " << entity.getCoordinates() << ", normals: " << entity.getNormals();
         EXPECT_EQ( grid.getEntityIndex( entity ), entity.getIndex() );
         if( entity.getIndex() >= 0 && entity.getIndex() < count )
            calls[ entity.getIndex() ]++;
         else
            ADD_FAILURE() << "Entity index " << entity.getIndex() << " is out of range [ 0, " << count << " ).";
      } );
   for( int i = 0; i < count; i++ )
      EXPECT_EQ( calls[ i ], 1 ) << "Entity index: " << i;
}
