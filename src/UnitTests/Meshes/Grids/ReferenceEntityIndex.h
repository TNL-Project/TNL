#pragma once

#include <TNL/Backend/Macros.h>

// Computes the index of the entity with given coordinates and normals independently of the indexing in the grid.
// The entities of one dimension are ordered by the orientation index and the entities of one orientation form
// the box [ 0, sizes + normals ), which is linearized with the x-coordinate changing the fastest.
template< typename Grid >
__cuda_callable__
typename Grid::IndexType
getReferenceEntityIndex(
   const Grid& grid,
   const typename Grid::CoordinatesType& coordinates,
   const typename Grid::CoordinatesType& normals )
{
   using Index = typename Grid::IndexType;
   constexpr int Dimension = Grid::getMeshDimension();
   int entityDimension = Dimension;
   for( int i = 0; i < Dimension; i++ )
      entityDimension -= normals[ i ];
   const auto& sizes = grid.getSizes();
   const int orientationsCount = Grid::getEntityOrientationsCount( entityDimension );
   Index offset = 0;
   for( int orientation = 0; orientation < orientationsCount; orientation++ ) {
      const auto orientationNormals = grid.getEntitiesOrientations().getNormals( entityDimension, orientation );
      Index count = 1;
      Index linearIndex = 0;
      for( int i = 0; i < Dimension; i++ ) {
         linearIndex += coordinates[ i ] * count;
         count *= sizes[ i ] + orientationNormals[ i ];
      }
      if( orientationNormals == normals )
         return offset + linearIndex;
      offset += count;
   }
   return -1;
}
