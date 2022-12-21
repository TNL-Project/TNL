#pragma once

#ifdef HAVE_GTEST

#include "Grid1DTestSuite.h"
#include <gtest/gtest.h>

TYPED_TEST_SUITE(GridTestSuite, Implementations);

TYPED_TEST(GridTestSuite, Test_0D_Neighbours_Of_0D_Entities_EntityIndexes )
{
   for (const auto& dimension : this->dimensions) {
      testNeighbourEntityIndexes<TypeParam, 0, 0>( this->grid, dimension );
   }
}

TYPED_TEST(GridTestSuite, Test_0D_Neighbours_Of_0D_Entities_DynamicGetter) {
   // EntityDimension | NeighbourEntityDimension | Orientation
   for (const auto& dimension : this->dimensions) {
      testNeighbourEntityIndexes<TypeParam, 0, 0>( this->grid, dimension );

      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 0, 0>(this -> grid, dimension);

      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 0, 0, 0>(this -> grid, dimension);
   }
}

TYPED_TEST(GridTestSuite, Test_1D_Neighbours_Of_0D_Entities_EntityIndexes )
{
   for (const auto& dimension : this->dimensions) {
      testNeighbourEntityIndexes<TypeParam, 0, 1>( this->grid, dimension );
   }
}

TYPED_TEST(GridTestSuite, Test_1D_Neighbours_Of_0D_Entities_DynamicGetter) {
   // EntityDimension | NeighbourEntityDimension | Orientation
   for (const auto& dimension : this->dimensions) {
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 0, 1>(this -> grid, dimension);

      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 0, 1, 0>(this -> grid, dimension);
   }
}

#endif

#include "../../../main.h"
