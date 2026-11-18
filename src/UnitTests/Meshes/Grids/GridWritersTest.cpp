#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Writers/detail/VTKMeshEntitiesWriter.h>
#include <TNL/Meshes/Writers/detail/VTUMeshEntitiesCollector.h>

// Returns the indexes of the vertices of the grid entity with the given index. The vertices are the corners of
// the entity, i.e. its coordinates shifted by all combinations of the basis vectors which the entity spans.
template< typename Grid, int EntityDimension >
std::set< int >
getEntityVertices( const Grid& grid, int entityIndex )
{
   using Entity = typename Grid::template EntityType< EntityDimension >;
   constexpr int Dimension = Grid::getMeshDimension();
   const Entity entity( grid, entityIndex );
   const auto basis = entity.getBasis();
   const auto& sizes = grid.getSizes();

   std::set< int > vertices;
   for( int shift = 0; shift < ( 1 << Dimension ); shift++ ) {
      bool valid = true;
      int vertexIndex = 0;
      int stride = 1;
      for( int i = 0; i < Dimension; i++ ) {
         const int bit = ( shift >> i ) & 1;
         if( bit && ! basis[ i ] )
            valid = false;
         vertexIndex += ( entity.getCoordinates()[ i ] + bit ) * stride;
         stride *= sizes[ i ] + 1;
      }
      if( valid )
         vertices.insert( vertexIndex );
   }
   return vertices;
}

// Checks that the VTK writer writes the grid entities in the order of their indexes in the grid.
template< typename Grid, int EntityDimension >
void
testVTKWriterEntitiesOrder( const Grid& grid )
{
   using Writer = TNL::Meshes::Writers::detail::VTKMeshEntitiesWriter< Grid, EntityDimension >;
   std::stringstream str;
   Writer::template writeConnectivity< int >( grid, str, TNL::Meshes::VTK::FileFormat::ascii );

   int entityIndex = 0;
   std::string line;
   while( std::getline( str, line ) ) {
      if( line.empty() )
         continue;
      std::stringstream lineStream( line );
      std::set< int > vertices;
      int vertex;
      while( lineStream >> vertex )
         vertices.insert( vertex );
      ASSERT_LT( entityIndex, grid.template getEntitiesCount< EntityDimension >() );
      EXPECT_EQ( vertices, ( getEntityVertices< Grid, EntityDimension >( grid, entityIndex ) ) )
         << "Grid dimension: " << Grid::getMeshDimension() << ", entity dimension: " << EntityDimension
         << ", entity index: " << entityIndex;
      entityIndex++;
   }
   EXPECT_EQ( entityIndex, grid.template getEntitiesCount< EntityDimension >() );
}

// Checks that the VTU collector collects the grid entities in the order of their indexes in the grid.
template< typename Grid, int EntityDimension >
void
testVTUCollectorEntitiesOrder( const Grid& grid )
{
   using Collector = TNL::Meshes::Writers::detail::MeshEntitiesVTUCollector< Grid, EntityDimension >;
   std::vector< typename Grid::GlobalIndexType > connectivity;
   std::vector< typename Grid::GlobalIndexType > offsets;
   std::vector< std::uint8_t > types;
   Collector::exec( grid, connectivity, offsets, types );

   ASSERT_EQ( (int) offsets.size(), grid.template getEntitiesCount< EntityDimension >() );
   std::size_t begin = 0;
   for( std::size_t entityIndex = 0; entityIndex < offsets.size(); entityIndex++ ) {
      const std::set< int > vertices( connectivity.begin() + begin, connectivity.begin() + offsets[ entityIndex ] );
      EXPECT_EQ( vertices, ( getEntityVertices< Grid, EntityDimension >( grid, entityIndex ) ) )
         << "Grid dimension: " << Grid::getMeshDimension() << ", entity dimension: " << EntityDimension
         << ", entity index: " << entityIndex;
      begin = offsets[ entityIndex ];
   }
}

template< int Dimension >
void
testWritersEntitiesOrder()
{
   using Grid = TNL::Meshes::Grid< Dimension, double, TNL::Devices::Host, int >;
   Grid grid;
   // different sizes along the axes, so that entities with different orientations have different counts
   typename Grid::CoordinatesType sizes;
   for( int i = 0; i < Dimension; i++ )
      sizes[ i ] = 2 + i;
   grid.setSizes( sizes );

   TNL::Algorithms::staticFor< int, 0, Dimension + 1 >(
      [ & ]( auto entityDimension )
      {
         constexpr int EntityDimension = decltype( entityDimension )::value;
         testVTKWriterEntitiesOrder< Grid, EntityDimension >( grid );
         testVTUCollectorEntitiesOrder< Grid, EntityDimension >( grid );
      } );
}

TEST( GridWritersTest, EntitiesOrder_1D )
{
   testWritersEntitiesOrder< 1 >();
}

TEST( GridWritersTest, EntitiesOrder_2D )
{
   testWritersEntitiesOrder< 2 >();
}

TEST( GridWritersTest, EntitiesOrder_3D )
{
   testWritersEntitiesOrder< 3 >();
}

#include "../../main.h"
