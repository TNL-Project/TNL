#pragma once

template<typename Grid, int EntityDimension>
void testGetEntityFromIndex( Grid& grid,
                             const typename Grid::CoordinatesType& dimensions,
                             const typename Grid::PointType& origin = typename Grid::PointType(0),
                             const typename Grid::PointType& spaceSteps = typename Grid::PointType(1)) {
   SCOPED_TRACE("Grid dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Grid dimensions: " + TNL::convertToString(dimensions));

   EXPECT_NO_THROW( grid.setDimensions( dimensions ) ) << "Verify, that the set of" << dimensions << " doesn't cause assert";
   EXPECT_NO_THROW( grid.setOrigin( origin ) ) << "Verify, that the set of" << origin << "doesn't cause assert";
   EXPECT_NO_THROW( grid.setSpaceSteps( spaceSteps ) ) << "Verify, that the set of" << spaceSteps << "doesn't cause assert";

   SCOPED_TRACE("Entity dimension: " + TNL::convertToString(EntityDimension));

   grid.template forAllEntities< EntityDimension >( [=] ( TNL::Meshes::GridEntity< Grid, EntityDimension >& entity ) mutable {
         auto new_entity = TNL::Meshes::GridEntity< Grid, EntityDimension >( entity.getGrid(), entity.getIndex() );
         EXPECT_EQ( new_entity.getCoordinates(), entity.getCoordinates() )
            << " Entity coordinates: " << entity.getCoordinates() << std::endl
            << " New entity coordinates: " << new_entity.getCoordinates() << std::endl;
         EXPECT_EQ( new_entity.getNormals(), entity.getNormals() )
            << " Entity normals: " << entity.getNormals() << std::endl
            << " New entity normals: " << new_entity.getNormals();
         EXPECT_EQ( new_entity.getOrientation().getOrientationIndex(), entity.getOrientation().getOrientationIndex() );
   } );
}
