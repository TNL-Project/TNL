#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

#include "ReferenceEntityIndex.h"

// The tests compare the adjacency functions of the grid with the indexes computed independently by
// getReferenceEntityIndex on grids with different sizes along the axes, so that swapped axes are detected.

template< int Dimension >
using GridType = TNL::Meshes::Grid< Dimension, double, TNL::Devices::Sequential, int >;

template< int Dimension >
std::vector< typename GridType< Dimension >::CoordinatesType >
getSizes()
{
   if constexpr( Dimension == 1 )
      return { { 1 }, { 5 } };
   if constexpr( Dimension == 2 )
      return { { 1, 1 }, { 3, 5 }, { 4, 1 }, { 1, 4 } };
   if constexpr( Dimension == 3 )
      return { { 1, 1, 1 }, { 2, 3, 4 }, { 4, 3, 2 }, { 3, 1, 2 } };
}

template< int Dimension >
typename GridType< Dimension >::CoordinatesType
getUnitVector( int axis )
{
   typename GridType< Dimension >::CoordinatesType result = 0;
   result[ axis ] = 1;
   return result;
}

// Checks the cells adjacent to all faces which do not lie on the boundary of the grid.
template< int Dimension >
void
testAdjacentCells( const typename GridType< Dimension >::CoordinatesType& sizes )
{
   using Grid = GridType< Dimension >;
   using Coordinates = typename Grid::CoordinatesType;
   using Face = typename Grid::template EntityType< Dimension - 1 >;
   SCOPED_TRACE( "Sizes: " + std::string( TNL::convertToString( sizes ).getString() ) );
   Grid grid;
   grid.setSizes( sizes );
   int checked = 0;
   TNL::Meshes::forAllEntities< Dimension - 1 >(
      grid,
      [ & ]( const Face& face )
      {
         // the face with the orientation index i has the normal along the axis i
         const int axis = face.getOrientationIndex();
      EXPECT_EQ( face.getNormals(), getUnitVector< Dimension >( axis ) );
         const Coordinates& coordinates = face.getCoordinates();
         if( coordinates[ axis ] == 0 || coordinates[ axis ] == sizes[ axis ] )
            return;
         int closer = -1;
         int remoter = -1;
         grid.getAdjacentCells( face, closer, remoter );
      EXPECT_EQ( closer, getReferenceEntityIndex( grid, coordinates - getUnitVector< Dimension >( axis ), Coordinates( 0 ) ) )
            << "Face coordinates: " << coordinates << ", normals: " << face.getNormals();
      EXPECT_EQ( remoter, getReferenceEntityIndex( grid, coordinates, Coordinates( 0 ) ) ) << "Face coordinates: " << coordinates
            << ", normals: " << face.getNormals();
         checked++;
      } );
   // the interior faces with the normal along the axis i form the box with sizes[ i ] - 1 positions along the axis i
   int expected = 0;
   for( int axis = 0; axis < Dimension; axis++ ) {
      int count = 1;
      for( int i = 0; i < Dimension; i++ )
         count *= i == axis ? sizes[ i ] - 1 : sizes[ i ];
      expected += count;
   }
   EXPECT_EQ( checked, expected );
}

// Checks the faces of all cells.
template< int Dimension >
void
testAdjacentFaces( const typename GridType< Dimension >::CoordinatesType& sizes )
{
   using Grid = GridType< Dimension >;
   using Coordinates = typename Grid::CoordinatesType;
   using Cell = typename Grid::template EntityType< Dimension >;
   SCOPED_TRACE( "Sizes: " + std::string( TNL::convertToString( sizes ).getString() ) );
   Grid grid;
   grid.setSizes( sizes );
   TNL::Meshes::forAllEntities< Dimension >(
      grid,
      [ & ]( const Cell& cell )
      {
         Coordinates closer = -1;
         Coordinates remoter = -1;
         grid.getAdjacentFacesIndexes( cell, closer, remoter );
         const Coordinates& coordinates = cell.getCoordinates();
         for( int axis = 0; axis < Dimension; axis++ ) {
            const Coordinates normals = getUnitVector< Dimension >( axis );
            EXPECT_EQ( closer[ axis ], getReferenceEntityIndex( grid, coordinates, normals ) )
               << "Cell coordinates: " << coordinates << ", axis: " << axis;
            EXPECT_EQ( remoter[ axis ], getReferenceEntityIndex( grid, coordinates + normals, normals ) )
               << "Cell coordinates: " << coordinates << ", axis: " << axis;
         }
      } );
}

// Checks the indexes of the adjacent entities with the same orientation in all directions.
template< int Dimension, int EntityDimension >
void
testAdjacentEntityIndex( const typename GridType< Dimension >::CoordinatesType& sizes )
{
   using Grid = GridType< Dimension >;
   using Coordinates = typename Grid::CoordinatesType;
   using Entity = typename Grid::template EntityType< EntityDimension >;
   SCOPED_TRACE(
      "Entity dimension: " + std::to_string( EntityDimension )
      + ", sizes: " + std::string( TNL::convertToString( sizes ).getString() ) );
   Grid grid;
   grid.setSizes( sizes );
   TNL::Meshes::forAllEntities< EntityDimension >(
      grid,
      [ & ]( const Entity& entity )
      {
         const Coordinates& normals = entity.getNormals();
         TNL::Algorithms::staticFor< int, 0, Dimension >(
            [ & ]( auto direction )
            {
               constexpr int Direction = decltype( direction )::value;
               auto check = [ & ]( int step, int index )
               {
                  const Coordinates neighbour = entity.getCoordinates() + step * getUnitVector< Dimension >( Direction );
                  if( neighbour[ Direction ] < 0 || neighbour[ Direction ] >= sizes[ Direction ] + normals[ Direction ] )
                     return;
                  EXPECT_EQ( index, getReferenceEntityIndex( grid, neighbour, normals ) )
                     << "Entity coordinates: " << entity.getCoordinates() << ", normals: " << normals
                     << ", direction: " << Direction << ", step: " << step;
               };
               check( -1, grid.template getAdjacentEntityIndex< Direction, -1 >( entity ) );
               check( 1, grid.template getAdjacentEntityIndex< Direction, 1 >( entity ) );
            } );
      } );
}

// Checks that the views returned by partitionEntities contain exactly the values of the entities with given
// orientation.
template< int Dimension, int EntityDimension >
void
testPartitionEntities( const typename GridType< Dimension >::CoordinatesType& sizes )
{
   using Grid = GridType< Dimension >;
   using Coordinates = typename Grid::CoordinatesType;
   SCOPED_TRACE(
      "Entity dimension: " + std::to_string( EntityDimension )
      + ", sizes: " + std::string( TNL::convertToString( sizes ).getString() ) );
   Grid grid;
   grid.setSizes( sizes );
   TNL::Containers::Vector< int, TNL::Devices::Sequential, int > values( grid.template getEntitiesCount< EntityDimension >() );
   for( int i = 0; i < values.getSize(); i++ )
      values[ i ] = i;
   const auto& constValues = values;
   int total = 0;
   for( int orientation = 0; orientation < Grid::getEntityOrientationsCount( EntityDimension ); orientation++ ) {
      const Coordinates normals = grid.template getNormals< EntityDimension >( orientation );
      int count = 1;
      for( int i = 0; i < Dimension; i++ )
         count *= sizes[ i ] + normals[ i ];
      auto view = grid.partitionEntities( values, EntityDimension, orientation );
      auto constView = grid.partitionEntities( constValues, EntityDimension, orientation );
      ASSERT_EQ( view.getSize(), count ) << "Orientation: " << orientation;
      ASSERT_EQ( constView.getSize(), count ) << "Orientation: " << orientation;
      EXPECT_EQ( view.getSize(), grid.getOrientedEntitiesCount( EntityDimension, orientation ) );
      const int first = getReferenceEntityIndex( grid, Coordinates( 0 ), normals );
      for( int i = 0; i < count; i++ ) {
         EXPECT_EQ( view[ i ], first + i ) << "Orientation: " << orientation;
         EXPECT_EQ( constView[ i ], first + i ) << "Orientation: " << orientation;
      }
      total += count;
   }
   EXPECT_EQ( total, values.getSize() );
}

// Checks the measure of the cells with different space steps along the axes.
template< int Dimension >
void
testCellMeasure( const typename GridType< Dimension >::CoordinatesType& sizes )
{
   using Grid = GridType< Dimension >;
   using Point = typename Grid::PointType;
   using Cell = typename Grid::template EntityType< Dimension >;
   Grid grid;
   grid.setSizes( sizes );
   Point origin;
   Point proportions;
   double expected = 1;
   for( int i = 0; i < Dimension; i++ ) {
      origin[ i ] = -1.0 - i;
      proportions[ i ] = 0.5 * ( i + 1 ) * sizes[ i ];
      expected *= 0.5 * ( i + 1 );
   }
   grid.setDomain( origin, proportions );
   EXPECT_DOUBLE_EQ( grid.getCellMeasure(), expected );
   TNL::Meshes::forAllEntities< Dimension >(
      grid,
      [ & ]( const Cell& cell )
      {
         EXPECT_DOUBLE_EQ( cell.getMeasure(), expected );
      } );
}

template< int Dimension >
void
testAdjacency()
{
   for( const auto& sizes : getSizes< Dimension >() ) {
      testAdjacentCells< Dimension >( sizes );
      testAdjacentFaces< Dimension >( sizes );
      testCellMeasure< Dimension >( sizes );
      TNL::Algorithms::staticFor< int, 0, Dimension + 1 >(
         [ & ]( auto entityDimension )
         {
            testAdjacentEntityIndex< Dimension, decltype( entityDimension )::value >( sizes );
            testPartitionEntities< Dimension, decltype( entityDimension )::value >( sizes );
         } );
   }
}

TEST( GridAdjacencyTest, Grid_1D )
{
   testAdjacency< 1 >();
}

TEST( GridAdjacencyTest, Grid_2D )
{
   testAdjacency< 2 >();
}

TEST( GridAdjacencyTest, Grid_3D )
{
   testAdjacency< 3 >();
}

#include "../../main.h"
