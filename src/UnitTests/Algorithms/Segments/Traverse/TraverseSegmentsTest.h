#pragma once

#include <gtest/gtest.h>

// Common test fixture template for all Traverse Segments tests
template< typename Segments >
class TraverseSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = Segments;
};

// Declare the typed test suite for parameterized tests
TYPED_TEST_SUITE_P( TraverseSegmentsTest );
