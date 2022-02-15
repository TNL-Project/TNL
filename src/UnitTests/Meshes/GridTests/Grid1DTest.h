
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
};

TYPED_TEST_SUITE(GridTestSuite, Implementations);

TYPED_TEST(GridTestSuite, TestSetWithParameterPack) {
   testIndexSet<TypeParam, true, 0>(this -> grid);
   testIndexSet<TypeParam, true, 1>(this -> grid);
   testIndexSet<TypeParam, true, 2>(this -> grid);
   testIndexSet<TypeParam, true, 11211>(this -> grid);
   testIndexSet<TypeParam, true, 232121>(this -> grid);
   testIndexSet<TypeParam, true, 434343>(this -> grid);

   testIndexSet<TypeParam, false, -1>(this -> grid);
   testIndexSet<TypeParam, false, -2>(this -> grid);
   testIndexSet<TypeParam, false, -3>(this -> grid);
   testIndexSet<TypeParam, false, -12312>(this -> grid);
   testIndexSet<TypeParam, false, -5454>(this -> grid);
   testIndexSet<TypeParam, false, -3424243>(this -> grid);
}

TYPED_TEST(GridTestSuite, TestSetWithCoordinates) {
   testContainerSet<TypeParam, true, 0>(this -> grid);
   testContainerSet<TypeParam, true, 1>(this -> grid);
   testContainerSet<TypeParam, true, 2>(this -> grid);
   testContainerSet<TypeParam, true, 10232>(this -> grid);
   testContainerSet<TypeParam, true, 45235423>(this -> grid);
   testContainerSet<TypeParam, true, 3231312>(this -> grid);

   testContainerSet<TypeParam, false, -1>(this -> grid);
   testContainerSet<TypeParam, false, -2>(this -> grid);
   testContainerSet<TypeParam, false, -3>(this -> grid);
   testContainerSet<TypeParam, false, -1232>(this -> grid);
   testContainerSet<TypeParam, false, -3243>(this -> grid);
   testContainerSet<TypeParam, false, -43121>(this -> grid);
}

#endif
