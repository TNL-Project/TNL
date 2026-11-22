#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/VTKTraits.h>
#include <TNL/Meshes/Writers/detail/VTKMeshEntitiesWriter.h>
#include <TNL/Meshes/Writers/detail/VTUMeshEntitiesCollector.h>

// Returns the indexes of the vertices of the grid entity with the given index in the order of the VTK cell types
// VTK_VERTEX, VTK_LINE, VTK_PIXEL and VTK_VOXEL. The vertices are the corners of the entity, i.e. its coordinates
// shifted by all combinations of the basis vectors which the entity spans, ordered with the x-axis changing
// the fastest. The indexes of the vertices are taken from the grid, not computed by the formula of the writers.
template< typename Grid, int EntityDimension >
std::vector< int >
getEntityVertices( const Grid& grid, int entityIndex )
{
   using Entity = typename Grid::template EntityType< EntityDimension >;
   using Vertex = typename Grid::template EntityType< 0 >;
   constexpr int Dimension = Grid::getMeshDimension();
   const Entity entity( grid, entityIndex );
   const auto basis = entity.getBasis();

   std::vector< int > vertices;
   for( int shift = 0; shift < ( 1 << Dimension ); shift++ ) {
      bool valid = true;
      typename Grid::CoordinatesType coordinates = entity.getCoordinates();
      for( int i = 0; i < Dimension; i++ ) {
         const int bit = ( shift >> i ) & 1;
         if( bit && ! basis[ i ] )
            valid = false;
         coordinates[ i ] += bit;
      }
      if( valid )
         vertices.push_back( Vertex( grid, coordinates ).getIndex() );
   }
   return vertices;
}

// Reads all integers written by the writer.
std::vector< int >
readValues( std::stringstream& str )
{
   std::vector< int > values;
   int value;
   while( str >> value )
      values.push_back( value );
   return values;
}

// Checks that the VTK writer writes the grid entities in the order of their indexes in the grid with the vertices
// in the order of the VTK cell types, and the offsets of the entities.
template< typename Grid, int EntityDimension >
void
testVTKWriter( const Grid& grid )
{
   using Writer = TNL::Meshes::Writers::detail::VTKMeshEntitiesWriter< Grid, EntityDimension >;
   const int count = grid.template getEntitiesCount< EntityDimension >();
   const int verticesPerEntity = 1 << EntityDimension;
   SCOPED_TRACE(
      "Grid dimension: " + std::to_string( Grid::getMeshDimension() )
      + ", entity dimension: " + std::to_string( EntityDimension ) );

   std::stringstream connectivityStream;
   Writer::template writeConnectivity< int >( grid, connectivityStream, TNL::Meshes::VTK::FileFormat::ascii );
   const std::vector< int > connectivity = readValues( connectivityStream );
   ASSERT_EQ( (int) connectivity.size(), count * verticesPerEntity );
   for( int entityIndex = 0; entityIndex < count; entityIndex++ ) {
      const std::vector< int > vertices(
         connectivity.begin() + entityIndex * verticesPerEntity,
         connectivity.begin() + ( entityIndex + 1 ) * verticesPerEntity );
      EXPECT_EQ( vertices, ( getEntityVertices< Grid, EntityDimension >( grid, entityIndex ) ) ) << "Entity index: " << entityIndex;
   }

   std::stringstream offsetsStream;
   Writer::template writeOffsets< int >( grid, offsetsStream, TNL::Meshes::VTK::FileFormat::ascii );
   const std::vector< int > offsets = readValues( offsetsStream );
   // the legacy VTK format starts the offsets with zero
   ASSERT_TRUE( (int) offsets.size() == count || (int) offsets.size() == count + 1 );
   const int first = (int) offsets.size() == count + 1 ? 0 : verticesPerEntity;
   for( std::size_t i = 0; i < offsets.size(); i++ )
      EXPECT_EQ( offsets[ i ], first + (int) i * verticesPerEntity ) << "Offset index: " << i;
}

// Checks that the VTU collector collects the grid entities in the order of their indexes in the grid with
// the vertices in the order of the VTK cell types, and the offsets and the types of the entities.
template< typename Grid, int EntityDimension >
void
testVTUCollector( const Grid& grid )
{
   using Collector = TNL::Meshes::Writers::detail::MeshEntitiesVTUCollector< Grid, EntityDimension >;
   const int count = grid.template getEntitiesCount< EntityDimension >();
   const int verticesPerEntity = 1 << EntityDimension;
   SCOPED_TRACE(
      "Grid dimension: " + std::to_string( Grid::getMeshDimension() )
      + ", entity dimension: " + std::to_string( EntityDimension ) );
   std::vector< typename Grid::GlobalIndexType > connectivity;
   std::vector< typename Grid::GlobalIndexType > offsets;
   std::vector< std::uint8_t > types;
   Collector::exec( grid, connectivity, offsets, types );

   const TNL::Meshes::VTK::EntityShape shapes[] = { TNL::Meshes::VTK::EntityShape::Vertex,
                                                    TNL::Meshes::VTK::EntityShape::Line,
                                                    TNL::Meshes::VTK::EntityShape::Pixel,
                                                    TNL::Meshes::VTK::EntityShape::Voxel };
   ASSERT_EQ( (int) offsets.size(), count );
   ASSERT_EQ( (int) types.size(), count );
   ASSERT_EQ( (int) connectivity.size(), count * verticesPerEntity );
   for( int entityIndex = 0; entityIndex < count; entityIndex++ ) {
      EXPECT_EQ( offsets[ entityIndex ], ( entityIndex + 1 ) * verticesPerEntity ) << "Entity index: " << entityIndex;
      EXPECT_EQ( types[ entityIndex ], (std::uint8_t) shapes[ EntityDimension ] ) << "Entity index: " << entityIndex;
      const std::vector< int > vertices(
         connectivity.begin() + entityIndex * verticesPerEntity,
         connectivity.begin() + ( entityIndex + 1 ) * verticesPerEntity );
      EXPECT_EQ( vertices, ( getEntityVertices< Grid, EntityDimension >( grid, entityIndex ) ) ) << "Entity index: " << entityIndex;
   }
}

template< int Dimension >
void
testWriters()
{
   using Grid = TNL::Meshes::Grid< Dimension, double, TNL::Devices::Host, int >;
   using Coordinates = typename Grid::CoordinatesType;
   // different sizes along the axes, so that entities with different orientations have different counts,
   // and grids with one cell along some axis
   std::vector< Coordinates > allSizes;
   Coordinates sizes;
   for( int i = 0; i < Dimension; i++ )
      sizes[ i ] = 2 + i;
   allSizes.push_back( sizes );
   allSizes.push_back( Coordinates( 1 ) );
   if constexpr( Dimension > 1 ) {
      sizes[ 0 ] = 1;
      allSizes.push_back( sizes );
   }

   for( const auto& gridSizes : allSizes ) {
      SCOPED_TRACE( "Sizes: " + std::string( TNL::convertToString( gridSizes ).getString() ) );
      Grid grid;
      grid.setSizes( gridSizes );
      TNL::Algorithms::staticFor< int, 0, Dimension + 1 >(
         [ & ]( auto entityDimension )
         {
            constexpr int EntityDimension = decltype( entityDimension )::value;
            testVTKWriter< Grid, EntityDimension >( grid );
            testVTUCollector< Grid, EntityDimension >( grid );
         } );
   }
}

TEST( GridWritersTest, Writers_1D )
{
   testWriters< 1 >();
}

TEST( GridWritersTest, Writers_2D )
{
   testWriters< 2 >();
}

TEST( GridWritersTest, Writers_3D )
{
   testWriters< 3 >();
}

#include "../../main.h"
