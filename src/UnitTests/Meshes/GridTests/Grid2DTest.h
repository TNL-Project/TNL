
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
};

TYPED_TEST_SUITE(GridTestSuite, Implementations);

TYPED_TEST(GridTestSuite, TestSetWithParameterPack) {
   testIndexSet<TypeParam, true, 0, 0>(this -> grid);
   testIndexSet<TypeParam, true, 0, 1>(this -> grid);
   testIndexSet<TypeParam, true, 0, 2>(this -> grid);
   testIndexSet<TypeParam, true, 1, 11211>(this -> grid);
   testIndexSet<TypeParam, true, 232121, 21>(this -> grid);
   testIndexSet<TypeParam, true, 54544, 434343>(this -> grid);

   testIndexSet<TypeParam, false, -1, 0>(this -> grid);
   testIndexSet<TypeParam, false, 0, -1>(this -> grid);
   testIndexSet<TypeParam, false, -2, -1>(this -> grid);
   testIndexSet<TypeParam, false, -2, 1>(this -> grid);
   testIndexSet<TypeParam, false, 2, -3>(this -> grid);
   testIndexSet<TypeParam, false, 3, -12312>(this -> grid);
   testIndexSet<TypeParam, false, 43, -5454>(this -> grid);
   testIndexSet<TypeParam, false, -3424243, 54234>(this -> grid);
}

TYPED_TEST(GridTestSuite, TestSetWithCoordinates) {
   testContainerSet<TypeParam, true, 0, 0>(this -> grid);
   testContainerSet<TypeParam, true, 0, 1>(this -> grid);
   testContainerSet<TypeParam, true, 0, 2>(this -> grid);
   testContainerSet<TypeParam, true, 1, 11211>(this -> grid);
   testContainerSet<TypeParam, true, 232121, 21>(this -> grid);
   testContainerSet<TypeParam, true, 54544, 434343>(this -> grid);

   testContainerSet<TypeParam, false, -1, 0>(this -> grid);
   testContainerSet<TypeParam, false, 0, -1>(this -> grid);
   testContainerSet<TypeParam, false, -2, -1>(this -> grid);
   testContainerSet<TypeParam, false, -2, 1>(this -> grid);
   testContainerSet<TypeParam, false, 2, -3>(this -> grid);
   testContainerSet<TypeParam, false, 3, -12312>(this -> grid);
   testContainerSet<TypeParam, false, 43, -5454>(this -> grid);
   testContainerSet<TypeParam, false, -3424243, 54234>(this -> grid);
}

#endif
