
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedGrid.h>

#include "support.h"

using Implementations = ::testing::Types<
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

TYPED_TEST(GridTestSuite, TestForAllTraverse_0D_Entity) {
   testForAllTraverse<TypeParam, 0>(this -> grid, { 1 });
   testForAllTraverse<TypeParam, 0>(this -> grid, { 2 });
   testForAllTraverse<TypeParam, 0>(this -> grid, { 4 });
   testForAllTraverse<TypeParam, 0>(this -> grid, { 8 });

   testForAllTraverse<TypeParam, 0>(this -> grid, { 9 });
   testForAllTraverse<TypeParam, 0>(this -> grid, { 127 });

   testForAllTraverse<TypeParam, 0>(this -> grid, { 1024 });
}

TYPED_TEST(GridTestSuite, TestForAllTraverse_1D_Entity) {
   testForAllTraverse<TypeParam, 1>(this -> grid, { 1 });
   testForAllTraverse<TypeParam, 1>(this -> grid, { 2 });
   testForAllTraverse<TypeParam, 1>(this -> grid, { 4 });
   testForAllTraverse<TypeParam, 1>(this -> grid, { 8 });

   testForAllTraverse<TypeParam, 1>(this -> grid, { 9 });
   testForAllTraverse<TypeParam, 1>(this -> grid, { 127 });

   testForAllTraverse<TypeParam, 1>(this -> grid, { 1024 });
}

#endif
