#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::Grid< 1, double, TNL::Devices::Sequential, int >,
   TNL::Meshes::Grid< 1, float, TNL::Devices::Sequential, int > >;

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

   for( const auto& sizes : std::vector< typename GridType::CoordinatesType >{ 10, 1 } )
      testGetEntityFromIndex< GridType, 0 >( grid, sizes );
}

TYPED_TEST( GridTestSuite, TestGetEntityFromIndex_1D_Entity )
{
   using GridType = typename TestFixture::GridType;

   GridType grid;

   for( const auto& sizes : std::vector< typename GridType::CoordinatesType >{ 10, 1 } )
      testGetEntityFromIndex< GridType, 1 >( grid, sizes );
}

#include "../../../main.h"
