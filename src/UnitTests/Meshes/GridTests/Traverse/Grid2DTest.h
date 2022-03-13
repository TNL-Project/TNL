
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedGrid.h>

#include "support.h"

using Implementations = ::testing::Types<
 //  TNL::Meshes::Grid<2, double, TNL::Devices::Host, int>,
   // TNL::Meshes::Grid<2, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<2, double, TNL::Devices::Cuda, int>
   // TNL::Meshes::Grid<2, float, TNL::Devices::Cuda, int>,
   // TNL::Meshes::DistributedGrid<2, double, TNL::Devices::Host, int>,
   // TNL::Meshes::DistributedGrid<2, float, TNL::Devices::Host, int>,
   // TNL::Meshes::DistributedGrid<2, double, TNL::Devices::Cuda, int>,
   // TNL::Meshes::DistributedGrid<2, float, TNL::Devices::Cuda, int>
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
   testForAllTraverse<TypeParam, 0>(this -> grid, { 1, 1 });

   // testForAllTraverse<TypeParam, 0>(this -> grid, { 2, 1 });
   // testForAllTraverse<TypeParam, 0>(this -> grid, { 1, 2 });

   // testForAllTraverse<TypeParam, 0>(this -> grid, { 100, 1 });
   // testForAllTraverse<TypeParam, 0>(this -> grid, { 1, 100 });

   // testForAllTraverse<TypeParam, 0>(this -> grid, { 100, 100 });
}

TYPED_TEST(GridTestSuite, TestForAllTraverse_1D_Entity) {
   // testForAllTraverse<TypeParam, 1>(this -> grid, { 1, 1 });

   // testForAllTraverse<TypeParam, 1>(this -> grid, { 2, 1 });
   // testForAllTraverse<TypeParam, 1>(this -> grid, { 1, 2 });

   // testForAllTraverse<TypeParam, 1>(this -> grid, { 100, 1 });
   // testForAllTraverse<TypeParam, 1>(this -> grid, { 1, 100 });

   // testForAllTraverse<TypeParam, 1>(this -> grid, { 100, 100 });
}

TYPED_TEST(GridTestSuite, TestForAllTraverse_2D_Entity) {
   // testForAllTraverse<TypeParam, 2>(this -> grid, { 1, 1 });

   // testForAllTraverse<TypeParam, 2>(this -> grid, { 2, 1 });
   // testForAllTraverse<TypeParam, 2>(this -> grid, { 1, 2 });

   // testForAllTraverse<TypeParam, 2>(this -> grid, { 100, 1 });
   // testForAllTraverse<TypeParam, 2>(this -> grid, { 1, 100 });

   // testForAllTraverse<TypeParam, 2>(this -> grid, { 100, 100 });
}

#endif
