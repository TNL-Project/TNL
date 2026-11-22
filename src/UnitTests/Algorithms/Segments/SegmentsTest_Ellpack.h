#pragma once

// Ellpack-specific tests, included in the generated SegmentsTest_Ellpack sources
// (see tnl_add_segments_tests in CMakeLists.txt)

#include "SegmentsTestVariants.h"
#include "SegmentsTestSuite.hpp"

// Ellpack-specific test fixture
template< typename Segments >
class EllpackSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

using EllpackSegmentsTypes =
   TNL::UnitTests::SegmentsTestTypes< TNL::UnitTests::EllpackVariants, TNL::UnitTests::TestDevice, int, long >;

TYPED_TEST_SUITE( EllpackSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes_EllpackOnly )
{
   test_setSegmentsSizes_EqualSizes_EllpackOnly< typename TestFixture::SegmentsType >();
}
