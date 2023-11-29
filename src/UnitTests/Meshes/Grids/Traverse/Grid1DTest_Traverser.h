#pragma once

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
#if defined( __CUDACC__ )
   TNL::Meshes::Grid< 1, double, TNL::Devices::Host, int >,
   TNL::Meshes::Grid< 1, float, TNL::Devices::Host, int >,
   TNL::Meshes::Grid< 1, double, TNL::Devices::Cuda, int >,
   TNL::Meshes::Grid< 1, float, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
   TNL::Meshes::Grid< 1, double, TNL::Devices::Host, int >,
   TNL::Meshes::Grid< 1, float, TNL::Devices::Host, int >,
   TNL::Meshes::Grid< 1, double, TNL::Devices::Hip, int >,
   TNL::Meshes::Grid< 1, float, TNL::Devices::Hip, int >
#else
   TNL::Meshes::Grid< 1, double, TNL::Devices::Host, int >,
   TNL::Meshes::Grid< 1, float, TNL::Devices::Host, int >
#endif
   >;

template< class GridType >
class GridTestSuite : public ::testing::Test
{
protected:
   GridType grid;

   std::vector< typename GridType::CoordinatesType > dimensions = { { 1 },
                                                                    { 2 },
                                                                    { 4 },
                                                                    { 8 },
                                                                    { 9 }
#if defined( __CUDACC__ ) || defined( __HIP__ ) || defined( HAVE_OPENMP )
                                                                    ,
                                                                    { 127 },
                                                                    { 1024 }
#endif
   };

   std::vector< typename GridType::PointType > origins = { { 1 }, { -10 }, { 0.1 }, { 1 }, { 1 }, { -2 } };

   std::vector< typename GridType::PointType > spaceSteps = { { 0.1 }, { 2 }, { 0.1 }, { 1 }, { 12 } };
};

TYPED_TEST_SUITE( GridTestSuite, Implementations );

TYPED_TEST( GridTestSuite, TestForAllTraverse_0D_Entity )
{
   for( const auto& dimension : this->dimensions )
      for( const auto& origin : this->origins )
         for( const auto& spaceStep : this->spaceSteps )
            testForAllTraverse< TypeParam, 0 >( this->grid, dimension, origin, spaceStep );
}

TYPED_TEST( GridTestSuite, TestForAllTraverse_1D_Entity )
{
   for( const auto& dimension : this->dimensions )
      for( const auto& origin : this->origins )
         for( const auto& spaceStep : this->spaceSteps )
            testForAllTraverse< TypeParam, 1 >( this->grid, dimension, origin, spaceStep );
}

TYPED_TEST( GridTestSuite, TestForInteriorTraverse_0D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testForInteriorTraverse< TypeParam, 0 >( this->grid, dimension );
}

TYPED_TEST( GridTestSuite, TestForInteriorTraverse_1D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testForInteriorTraverse< TypeParam, 1 >( this->grid, dimension );
}

TYPED_TEST( GridTestSuite, TestForBoundaryTraverse_0D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testForBoundaryTraverse< TypeParam, 0 >( this->grid, dimension );
}

TYPED_TEST( GridTestSuite, TestForBoundaryTraverse_1D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testForBoundaryTraverse< TypeParam, 1 >( this->grid, dimension );
}

TYPED_TEST( GridTestSuite, TestBoundaryUnionInteriorEqualAllProperty_0D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testBoundaryUnionInteriorEqualAllProperty< TypeParam, 0 >( this->grid, dimension );
}

TYPED_TEST( GridTestSuite, TestBoundaryUnionInteriorEqualAllProperty_1D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testBoundaryUnionInteriorEqualAllProperty< TypeParam, 1 >( this->grid, dimension );
}

TYPED_TEST( GridTestSuite, TestAllMinusBoundaryEqualInteriorProperty_0D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testAllMinusBoundaryEqualInteriorProperty< TypeParam, 0 >( this->grid, dimension );
}

TYPED_TEST( GridTestSuite, TestAllMinusBoundaryEqualInteriorProperty_1D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testAllMinusBoundaryEqualInteriorProperty< TypeParam, 1 >( this->grid, dimension );
}

TYPED_TEST( GridTestSuite, TestAllMinusInteriorEqualBoundaryProperty_0D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testAllMinusInteriorEqualBoundaryProperty< TypeParam, 0 >( this->grid, dimension );
}

TYPED_TEST( GridTestSuite, TestAllMinusInteriorEqualBoundaryProperty_1D_Entity )
{
   for( const auto& dimension : this->dimensions )
      testAllMinusInteriorEqualBoundaryProperty< TypeParam, 1 >( this->grid, dimension );
}

#include "../../../main.h"
