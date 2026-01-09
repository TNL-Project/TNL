#pragma once

#include <gtest/gtest.h>

// Common test fixture template for all Segments tests
template< typename Segments >
class SegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// Declare the typed test suite for parameterized tests
TYPED_TEST_SUITE_P( SegmentsTest );
