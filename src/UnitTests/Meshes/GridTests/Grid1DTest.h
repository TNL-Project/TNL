
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedGrid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::NDimGrid<1, double, TNL::Devices::Host, int>,
   TNL::Meshes::NDimGrid<1, float, TNL::Devices::Host, int>,
   TNL::Meshes::NDimGrid<1, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::NDimGrid<1, float, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<1, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<1, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<1, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<1, float, TNL::Devices::Cuda, int>,
   TNL::Meshes::DistributedGrid<1, double, TNL::Devices::Host, int>,
   TNL::Meshes::DistributedGrid<1, float, TNL::Devices::Host, int>,
   TNL::Meshes::DistributedGrid<1, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::DistributedGrid<1, float, TNL::Devices::Cuda, int>
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
   EXPECT_EQ(TypeParam::getMeshDimension(), 1) << "All grids must have dimension 1";
}

TYPED_TEST(GridTestSuite, TestSetWithParameterPack) {
   testDimensionSetByIndex<TypeParam, true, 0>(this->grid);
   testDimensionSetByIndex<TypeParam, true, 1>(this->grid);
   testDimensionSetByIndex<TypeParam, true, 2>(this->grid);
   testDimensionSetByIndex<TypeParam, true, 11211>(this->grid);
   testDimensionSetByIndex<TypeParam, true, 232121>(this->grid);
   testDimensionSetByIndex<TypeParam, true, 434343>(this->grid);

   testDimensionSetByIndex<TypeParam, false, -1>(this->grid);
   testDimensionSetByIndex<TypeParam, false, -2>(this->grid);
   testDimensionSetByIndex<TypeParam, false, -3>(this->grid);
   testDimensionSetByIndex<TypeParam, false, -12312>(this->grid);
   testDimensionSetByIndex<TypeParam, false, -5454>(this->grid);
   testDimensionSetByIndex<TypeParam, false, -3424243>(this->grid);
}

TYPED_TEST(GridTestSuite, TestSetWithCoordinates) {
   testDimensionSetByCoordinate<TypeParam, true, 0>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 1>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 2>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 10232>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 45235423>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, true, 3231312>(this -> grid);

   testDimensionSetByCoordinate<TypeParam, false, -1>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, -2>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, -3>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, -1232>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, -3243>(this -> grid);
   testDimensionSetByCoordinate<TypeParam, false, -43121>(this -> grid);
}

TYPED_TEST(GridTestSuite, TestEntitiesCount) {
   // GridType, Edges, Vertices | Edges
   TestEntitiesCount<TypeParam, IntPack<0>, IntPack<0, 0>>::exec(this -> grid);
   TestEntitiesCount<TypeParam, IntPack<1>, IntPack<2, 1>>::exec(this -> grid);
   TestEntitiesCount<TypeParam, IntPack<2>, IntPack<3, 2>>::exec(this -> grid);

   TestEntitiesCount<TypeParam, IntPack<100>, IntPack<101, 100>>::exec(this -> grid);
}

TYPED_TEST(GridTestSuite, TestOriginSet) {
   testOriginSetByCoordinate<TypeParam>(this -> grid, 0.6);
   testOriginSetByCoordinate<TypeParam>(this -> grid, -1);
   testOriginSetByCoordinate<TypeParam>(this -> grid, 100);
   testOriginSetByCoordinate<TypeParam>(this -> grid, -100000);
   testOriginSetByCoordinate<TypeParam>(this -> grid, 323121);

   testOriginSetByIndex<TypeParam>(this -> grid, 0.6);
   testOriginSetByIndex<TypeParam>(this -> grid, -1);
   testOriginSetByIndex<TypeParam>(this -> grid, 100);
   testOriginSetByIndex<TypeParam>(this -> grid, -100000);
   testOriginSetByIndex<TypeParam>(this -> grid, 323121);
}

#endif
