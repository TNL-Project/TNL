#pragma once

#include <gtest/gtest.h>

// Common test fixture template for all Reduce Segments tests
template< typename Segments >
class ReduceSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// Declare the typed test suite for parameterized tests
TYPED_TEST_SUITE_P( ReduceSegmentsTest );
