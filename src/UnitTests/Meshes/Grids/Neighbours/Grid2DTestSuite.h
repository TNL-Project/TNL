
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::Grid<2, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<2, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<2, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<2, float, TNL::Devices::Cuda, int>
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
         { 3, 3 }
#if HAVE_CUDA || HAVE_OPENMP
         ,
         { 100, 1 },
         { 1, 100 },
         { 100, 100 }
#endif
      };

#ifndef HAVE_CUDA
      void SetUp() override {
         if (std::is_same<typename GridType::DeviceType, TNL::Devices::Cuda>::value) {
            GTEST_SKIP() << "No CUDA available on host. Try to compile with CUDA instead";
         }
      }
#endif
};


template<typename Grid, int EntityDimension, int NeighbourEntityDimension>
void testStaticNeighbourEntityGetterForAllStencils(Grid& grid, const typename Grid::CoordinatesType& dimension) {
   auto firstLoop = [&](const auto i){
      auto secondLoop = [&](const auto i, const auto j) {
         testStaticNeighbourEntityGetter<Grid, EntityDimension, NeighbourEntityDimension, i - 1, j - 1>(grid, dimension);
      };

      TNL::Algorithms::staticFor< int, 0, 3 >(secondLoop, i);
   };

   TNL::Algorithms::staticFor< int, 0, 3 >(firstLoop);
}

template<typename Grid, int EntityDimension, int NeighbourEntityDimension, int NeighbourEntityOrientation>
void testStaticNeighbourEntityGetterForAllStencils(Grid& grid, const typename Grid::CoordinatesType& dimension) {
   auto firstLoop = [&](const auto i){
      auto secondLoop = [&](const auto i, const auto j) {
         testStaticNeighbourEntityGetter<Grid, EntityDimension, NeighbourEntityDimension, i - 1, j - 1>(grid, dimension);
      };

      TNL::Algorithms::staticFor< int, 0, 3 >(secondLoop, i);
   };

   TNL::Algorithms::staticFor< int, 0, 3 >(firstLoop);
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
