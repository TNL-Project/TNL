#pragma once

#include "Grid2DTestSuite.h"
#include <gtest/gtest.h>

TYPED_TEST_SUITE( GridTestSuite, Implementations );

TYPED_TEST(GridTestSuite, Test_0D_Neighbours_Of_1D_Entities_EntityIndexes )
{
   for (const auto& dimension : this->dimensions) {
      testNeighbourEntityIndexes<TypeParam, 1, 0>( this->grid, dimension );
   }
}


TYPED_TEST(GridTestSuite, Test_0D_Neighbours_Of_1D_Entities_DynamicGetter) {
   for (const auto& dimension : this->dimensions) {
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 1, 0>(this -> grid, dimension);
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 1, 0, 0>(this -> grid, dimension);
   }
}

TYPED_TEST(GridTestSuite, Test_1D_Neighbours_Of_1D_Entities_EntityIndexes )
{
   for (const auto& dimension : this->dimensions) {
      testNeighbourEntityIndexes<TypeParam, 1, 1>( this->grid, dimension );
   }
}

TYPED_TEST(GridTestSuite, Test_1D_Neighbours_Of_1D_Entities_DynamicGetter) {
   for (const auto& dimension : this->dimensions) {
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 1, 1>(this -> grid, dimension);

      testDynamicNeighbourEntityGetterForAllStencils< TypeParam, 1, 1, 0 >( this->grid, dimension );
      testDynamicNeighbourEntityGetterForAllStencils< TypeParam, 1, 1, 1 >( this->grid, dimension );
   }
}

TYPED_TEST(GridTestSuite, Test_2D_Neighbours_Of_1D_Entities_EntityIndexes )
{
   for (const auto& dimension : this->dimensions) {
      testNeighbourEntityIndexes<TypeParam, 1, 2>( this->grid, dimension );
   }
}

TYPED_TEST(GridTestSuite, Test_2D_Neighbours_Of_1D_Entities_DynamicGetter) {
   for (const auto& dimension : this->dimensions) {
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 1, 2>(this -> grid, dimension);

      testDynamicNeighbourEntityGetterForAllStencils< TypeParam, 1, 2, 0 >( this->grid, dimension );
   }
}

#include "../../../main.h"
