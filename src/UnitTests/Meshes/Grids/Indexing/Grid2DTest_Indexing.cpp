#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::Grid< 2, double, TNL::Devices::Sequential, int >,
   TNL::Meshes::Grid< 2, float, TNL::Devices::Sequential, int > >;

template< typename Grid >
class GridTestSuite : public ::testing::Test
{
protected:
   using GridType = Grid;
};

TYPED_TEST_SUITE( GridTestSuite, Implementations );

TYPED_TEST( GridTestSuite, TestGetEntityFromIndex_0D_Entity )
{
   using GridType = typename TestFixture::GridType;

   GridType grid;

   for( const auto& sizes :
        std::vector< typename GridType::CoordinatesType >{ { 7, 7 }, { 3, 5 }, { 5, 3 }, { 1, 4 }, { 4, 1 } } )
      testGetEntityFromIndex< GridType, 0 >( grid, sizes );
}

TYPED_TEST( GridTestSuite, TestGetEntityFromIndex_1D_Entity )
{
   using GridType = typename TestFixture::GridType;

   GridType grid;

   for( const auto& sizes :
        std::vector< typename GridType::CoordinatesType >{ { 7, 7 }, { 3, 5 }, { 5, 3 }, { 1, 4 }, { 4, 1 } } )
      testGetEntityFromIndex< GridType, 1 >( grid, sizes );
}

TYPED_TEST( GridTestSuite, TestGetEntityFromIndex_2D_Entity )
{
   using GridType = typename TestFixture::GridType;

   GridType grid;

   for( const auto& sizes :
        std::vector< typename GridType::CoordinatesType >{ { 7, 7 }, { 3, 5 }, { 5, 3 }, { 1, 4 }, { 4, 1 } } )
      testGetEntityFromIndex< GridType, 2 >( grid, sizes );
}

#include "../../../main.h"
