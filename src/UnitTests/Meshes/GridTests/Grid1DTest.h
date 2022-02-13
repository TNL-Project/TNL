
#ifdef HAVE_GTEST
#include <TNL/Meshes/Grid.h>
#include <gtest/gtest.h>

using Implementations = ::testing::Types<
   TNL::Meshes::NDimGrid<1, double, TNL::Devices::Host, int>,
   TNL::Meshes::NDimGrid<1, float, TNL::Devices::Host, int>,
   TNL::Meshes::NDimGrid<1, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::NDimGrid<1, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<1, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<1, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<1, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<1, float, TNL::Devices::Host, int>
>;

template <class GridType>
class GridTestSuite: public ::testing::Test {
   protected:
      GridType grid;
};

TYPED_TEST_SUITE(GridTestSuite, Implementations);

TYPED_TEST(GridTestSuite, TestSetDimensions) {}

#endif
