
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedGrid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::NDimGrid<2, double, TNL::Devices::Host, int>,
   TNL::Meshes::NDimGrid<2, float, TNL::Devices::Host, int>,
   TNL::Meshes::NDimGrid<2, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::NDimGrid<2, float, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<2, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<2, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<2, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<2, float, TNL::Devices::Cuda, int>,
   TNL::Meshes::DistributedGrid<2, double, TNL::Devices::Host, int>,
   TNL::Meshes::DistributedGrid<2, float, TNL::Devices::Host, int>,
   TNL::Meshes::DistributedGrid<2, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::DistributedGrid<2, float, TNL::Devices::Cuda, int>
>;

template <class GridType>
class GridTestSuite: public ::testing::Test {
   protected:
      GridType grid;

#ifndef HAVE_CUDA
      void SetUp() override {
         if (std::is_same<typename GridType::DeviceType, TNL::Devices::Cuda>::value) {
            GTEST_SKIP() << "No CUDA available on host. Try to compile with CUDA instead";
         }
      }
#endif
};

TYPED_TEST_SUITE(GridTestSuite, Implementations);

TYPED_TEST(GridTestSuite, TestMeshDimensionGetter) {
   EXPECT_EQ(TypeParam::getMeshDimension(), 2) << "All grids must have dimension 2";
}

TYPED_TEST(GridTestSuite, TestSetWithParameterPack) {
   testDimensionSetByIndex<TypeParam, true, 0, 0>(this -> grid);
   testDimensionSetByIndex<TypeParam, true, 0, 1>(this -> grid);
   testDimensionSetByIndex<TypeParam, true, 0, 2>(this -> grid);
   testDimensionSetByIndex<TypeParam, true, 1, 11211>(this -> grid);
   testDimensionSetByIndex<TypeParam, true, 232121, 21>(this -> grid);
   testDimensionSetByIndex<TypeParam, true, 54544, 434343>(this -> grid);

   testDimensionSetByIndex<TypeParam, false, -1, 0>(this -> grid);
   testDimensionSetByIndex<TypeParam, false, 0, -1>(this -> grid);
   testDimensionSetByIndex<TypeParam, false, -2, -1>(this -> grid);
   testDimensionSetByIndex<TypeParam, false, -2, 1>(this -> grid);
   testDimensionSetByIndex<TypeParam, false, 2, -3>(this -> grid);
   testDimensionSetByIndex<TypeParam, false, 3, -12312>(this -> grid);
   testDimensionSetByIndex<TypeParam, false, 43, -5454>(this -> grid);
   testDimensionSetByIndex<TypeParam, false, -3424243, 54234>(this -> grid);
}

TYPED_TEST(GridTestSuite, TestSetWithCoordinates) {
   testDimensionSetByCoordinate<TypeParam, true, 0, 0>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 0, 1>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 0, 2>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 1, 11211>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 232121, 21>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 54544, 434343>(this -> grid);

   testDimensionSetByCoordinate<TypeParam, false, -1, 0>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, 0, -1>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, -2, -1>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, -2, 1>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, 2, -3>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, 3, -12312>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, 43, -5454>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, -3424243, 54234>(this -> grid);
}

TYPED_TEST(GridTestSuite, TestEntitiesCount) {
   // GridType, Edges, Vertices | Edges
   TestEntitiesCount<TypeParam, IntPack<0, 0>, IntPack<0, 0, 0>>::exec(this -> grid);

   TestEntitiesCount<TypeParam, IntPack<1, 0>, IntPack<0, 0, 0>>::exec(this -> grid);
   TestEntitiesCount<TypeParam, IntPack<0, 1>, IntPack<0, 0, 0>>::exec(this -> grid);

   TestEntitiesCount<TypeParam, IntPack<1, 1>, IntPack<4, 4, 1>>::exec(this -> grid);

   TestEntitiesCount<TypeParam, IntPack<2, 1>, IntPack<6, 7, 2>>::exec(this -> grid);
   TestEntitiesCount<TypeParam, IntPack<1, 2>, IntPack<6, 7, 2>>::exec(this -> grid);

   TestEntitiesCount<TypeParam, IntPack<2, 2>, IntPack<9, 12, 4>>::exec(this -> grid);

   TestEntitiesCount<TypeParam, IntPack<2, 3>, IntPack<12, 17, 6>>::exec(this -> grid);
   TestEntitiesCount<TypeParam, IntPack<3, 2>, IntPack<12, 17, 6>>::exec(this -> grid);

   TestEntitiesCount<TypeParam, IntPack<50, 50>, IntPack<51 * 51, 51 * 50 + 50 * 51, 50 * 50>>::exec(this -> grid);
   TestEntitiesCount<TypeParam, IntPack<50, 100>, IntPack<51 * 101, 51 * 100 + 50 * 101, 50 * 100>>::exec(this -> grid);
}

#endif
