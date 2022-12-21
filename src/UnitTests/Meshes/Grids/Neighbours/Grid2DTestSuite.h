
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::Grid<2, double, TNL::Devices::Sequential, int>,
   TNL::Meshes::Grid<2, float, TNL::Devices::Sequential, int>
#ifdef HAVE_CUDA
  ,TNL::Meshes::Grid<2, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<2, float, TNL::Devices::Cuda, int>
#endif
>;

template <class GridType>
class GridTestSuite: public ::testing::Test {
   protected:
      GridType grid;

      std::vector< typename GridType::CoordinatesType > dimensions = {
         { 1, 1 },
         { 2, 1 },
         { 1, 2 },
         { 2, 2 },
         { 3, 3 },
         { 1, 7 },
         { 7, 1 },
         { 7, 7 },
         { 3, 8 },
         { 8, 3 },
         { 8, 8 }
      };
};

template<typename Grid, int EntityDimension, int NeighbourEntityDimension>
void testNeighbourEntityIndexes(Grid& grid, const typename Grid::CoordinatesType& dimension, const int scale = 1) {
   for (int i = -1 * scale; i < 1 * scale; i++)
      testNeighbourEntityIndexes<Grid, EntityDimension, NeighbourEntityDimension>(grid, dimension, typename Grid::CoordinatesType(i));
}

template<typename Grid, int EntityDimension, int NeighbourEntityDimension>
void testDynamicNeighbourEntityGetterForAllStencils(Grid& grid, const typename Grid::CoordinatesType& dimension, const int scale = 1) {
   for (int i = -1 * scale; i < 1 * scale; i++)
      for (int j = -1 * scale; j < 1 * scale; j++)
         testDynamicNeighbourEntityGetter<Grid, EntityDimension, NeighbourEntityDimension>(grid, dimension, typename Grid::CoordinatesType(i, j));
}

template<typename Grid, int EntityDimension, int NeighbourEntityDimension, int NeighbourEntityOrientation>
void testDynamicNeighbourEntityGetterForAllStencils(Grid& grid, const typename Grid::CoordinatesType& dimension, const int scale = 1) {
   for (int i = -1 * scale; i < 1 * scale; i++)
      for (int j = -1 * scale; j < 1 * scale; j++)
         testDynamicNeighbourEntityGetter<Grid, EntityDimension, NeighbourEntityDimension, NeighbourEntityOrientation>(grid, dimension, typename Grid::CoordinatesType(i, j));
}

#endif
