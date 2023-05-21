#pragma once

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::Grid<3, double, TNL::Devices::Sequential, int>,
   TNL::Meshes::Grid<3, float, TNL::Devices::Sequential, int>
#ifdef __CUDACC__
  ,TNL::Meshes::Grid<3, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<3, float, TNL::Devices::Cuda, int>
#endif
                                          >;

template< class GridType >
class GridTestSuite : public ::testing::Test
{
protected:
   GridType grid;

   std::vector< typename GridType::CoordinatesType > dimensions = {
      { 1, 1, 1 }, { 2, 1, 1 }, { 1, 2, 1 }, { 1, 1, 2 }, { 2, 2, 2 }, { 5, 5, 5 }, { 5, 1, 1 }, { 1, 5, 1 }, { 1, 1, 5 },
      { 7, 9, 1 }, { 7, 1, 9 }, { 9, 7, 1 }, { 1, 7, 9 }, { 9, 1, 7 }, { 1, 9, 7 }, { 7, 9, 9 }, { 9, 7, 9 }, { 9, 9, 7 }
   };
};

template< typename Grid, int EntityDimension, int NeighbourEntityDimension >
void
testNeighbourEntityIndexes( Grid& grid, const typename Grid::CoordinatesType& dimension, const int scale = 1 )
{
   for( int i = -1 * scale; i < 1 * scale; i++ )
      testNeighbourEntityIndexes< Grid, EntityDimension, NeighbourEntityDimension >(
         grid, dimension, typename Grid::CoordinatesType( i ) );
}

template< typename Grid, int EntityDimension, int NeighbourEntityDimension >
void
testDynamicNeighbourEntityGetterForAllStencils( Grid& grid,
                                                const typename Grid::CoordinatesType& dimension,
                                                const int scale = 1 )
{
   for( int i = -1 * scale; i < 1 * scale; i++ )
      for( int j = -1 * scale; j < 1 * scale; j++ )
         for( int k = -1 * scale; k < 1 * scale; k++ )
            testDynamicNeighbourEntityGetter< Grid, EntityDimension, NeighbourEntityDimension >(
               grid, dimension, typename Grid::CoordinatesType( i, j, k ) );
}

template< typename Grid, int EntityDimension, int NeighbourEntityDimension, int NeighbourEntityOrientation >
void
testDynamicNeighbourEntityGetterForAllStencils( Grid& grid,
                                                const typename Grid::CoordinatesType& dimension,
                                                const int scale = 1 )
{
   for( int i = -1 * scale; i < 1 * scale; i++ )
      for( int j = -1 * scale; j < 1 * scale; j++ )
         for( int k = -1 * scale; k < 1 * scale; k++ )
            testDynamicNeighbourEntityGetter< Grid, EntityDimension, NeighbourEntityDimension, NeighbourEntityOrientation >(
               grid, dimension, typename Grid::CoordinatesType( i, j, k ) );
}
