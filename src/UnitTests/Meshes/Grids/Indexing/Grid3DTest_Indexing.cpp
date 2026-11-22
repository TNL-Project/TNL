#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::Grid< 3, double, TNL::Devices::Sequential, int >,
   TNL::Meshes::Grid< 3, float, TNL::Devices::Sequential, int > >;

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
        std::vector< typename GridType::CoordinatesType >{ { 3, 3, 3 }, { 2, 3, 4 }, { 4, 2, 3 }, { 1, 3, 2 }, { 1, 1, 1 } } )
      testGetEntityFromIndex< GridType, 0 >( grid, sizes );
}

TYPED_TEST( GridTestSuite, TestGetEntityFromIndex_1D_Entity )
{
   using GridType = typename TestFixture::GridType;

   GridType grid;

   for( const auto& sizes :
        std::vector< typename GridType::CoordinatesType >{ { 3, 3, 3 }, { 2, 3, 4 }, { 4, 2, 3 }, { 1, 3, 2 }, { 1, 1, 1 } } )
      testGetEntityFromIndex< GridType, 1 >( grid, sizes );
}

TYPED_TEST( GridTestSuite, TestGetEntityFromIndex_2D_Entity )
{
   using GridType = typename TestFixture::GridType;

   GridType grid;

   for( const auto& sizes :
        std::vector< typename GridType::CoordinatesType >{ { 3, 3, 3 }, { 2, 3, 4 }, { 4, 2, 3 }, { 1, 3, 2 }, { 1, 1, 1 } } )
      testGetEntityFromIndex< GridType, 2 >( grid, sizes );
}

TYPED_TEST( GridTestSuite, TestGetEntityFromIndex_3D_Entity )
{
   using GridType = typename TestFixture::GridType;

   GridType grid;

   for( const auto& sizes :
        std::vector< typename GridType::CoordinatesType >{ { 3, 3, 3 }, { 2, 3, 4 }, { 4, 2, 3 }, { 1, 3, 2 }, { 1, 1, 1 } } )
      testGetEntityFromIndex< GridType, 3 >( grid, sizes );
}

#include "../../../main.h"
