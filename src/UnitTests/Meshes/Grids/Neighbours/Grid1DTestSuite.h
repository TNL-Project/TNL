#pragma once

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::Grid< 1, double, TNL::Devices::Sequential, int >,
   TNL::Meshes::Grid< 1, float, TNL::Devices::Sequential, int >
#if defined( __CUDACC__ )
   ,
   TNL::Meshes::Grid< 1, double, TNL::Devices::Cuda, int >,
   TNL::Meshes::Grid< 1, float, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
   ,
   TNL::Meshes::Grid< 1, double, TNL::Devices::Hip, int >,
   TNL::Meshes::Grid< 1, float, TNL::Devices::Hip, int >
#endif
   >;

template< class GridType >
class GridTestSuite : public ::testing::Test
{
protected:
   GridType grid;

   std::vector< typename GridType::CoordinatesType > dimensions = { { 1 }, { 2 }, { 4 }, { 8 }, { 9 }, { 13 }, { 16 }, { 17 } };
};

template< typename Grid, int EntityDimension, int NeighbourEntityDimension >
void
testNeighbourEntityIndexes( Grid& grid, const typename Grid::CoordinatesType& dimension, const int scale = 1 )
{
   if constexpr( TNL::Devices::isSequential< typename Grid::DeviceType >() )
      for( int i = -scale; i <= scale; i++ )
         testNeighbourEntityIndexes< Grid, EntityDimension, NeighbourEntityDimension >(
            grid, dimension, typename Grid::CoordinatesType( i ) );
}

template< typename Grid, int EntityDimension, int NeighbourEntityDimension >
void
testDynamicNeighbourEntityGetterForAllStencils(
   Grid& grid,
   const typename Grid::CoordinatesType& dimension,
   const int scale = 1 )
{
   for( int i = -scale; i <= scale; i++ )
      testDynamicNeighbourEntityGetter< Grid, EntityDimension, NeighbourEntityDimension >(
         grid, dimension, typename Grid::CoordinatesType( i ) );
}

template< typename Grid, int EntityDimension, int NeighbourEntityDimension, int NeighbourEntityOrientation >
void
testDynamicNeighbourEntityGetterForAllStencils(
   Grid& grid,
   const typename Grid::CoordinatesType& dimension,
   const int scale = 1 )
{
   for( int i = -scale; i <= scale; i++ )
      testDynamicNeighbourEntityGetter< Grid, EntityDimension, NeighbourEntityDimension, NeighbourEntityOrientation >(
         grid, dimension, typename Grid::CoordinatesType( i ) );
}
