#pragma once

#include <TNL/Algorithms/Segments/SortedSegments.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <iostream>
#include <sstream>

#include "SparseMatrixTest.hpp"

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

TYPED_TEST_SUITE_P( MatrixTest );

TYPED_TEST_P( MatrixTest, Constructors )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_Constructors< MatrixType >();
}

TYPED_TEST_P( MatrixTest, setDimensionsTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetDimensions< MatrixType >();
}

TYPED_TEST_P( MatrixTest, setRowCapacitiesTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetRowCapacities< MatrixType >();
}

TYPED_TEST_P( MatrixTest, setLikeTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetLike< MatrixType, MatrixType >();
}

TYPED_TEST_P( MatrixTest, setElementsTest )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;

   if constexpr(
      std::is_same_v< DeviceType, TNL::Devices::Sequential > || std::is_same_v< DeviceType, TNL::Devices::Host >
      || (std::is_same_v< std::decay_t< RealType >, float > || std::is_same_v< std::decay_t< RealType >, double > || std::is_same_v< std::decay_t< RealType >, int > || std::is_same_v< std::decay_t< RealType >, long long int > || std::is_same_v< std::decay_t< RealType >, bool >) )
      test_SetElementsForSymmetricMatrix< MatrixType >();
}

TYPED_TEST_P( MatrixTest, setElementsFromVectorTest )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;

   // the test creates a symmetric matrix which requires atomic operations on GPU
   if constexpr(
      ! MatrixType::isSymmetric()
      && ( std::is_same_v< DeviceType, TNL::Devices::Sequential > || std::is_same_v< DeviceType, TNL::Devices::Host > || (std::is_same_v< std::decay_t< RealType >, float > || std::is_same_v< std::decay_t< RealType >, double > || std::is_same_v< std::decay_t< RealType >, int > || std::is_same_v< std::decay_t< RealType >, long long int > || std::is_same_v< std::decay_t< RealType >, bool >) ) )
      test_SetElementsFromVector< MatrixType >();
}

TYPED_TEST_P( MatrixTest, resetTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_Reset< MatrixType >();
}

TYPED_TEST_P( MatrixTest, getRowTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_GetRow< MatrixType >();
}

TYPED_TEST_P( MatrixTest, setElementTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SetElement< MatrixType >();
}

TYPED_TEST_P( MatrixTest, addElementTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_AddElement< MatrixType >();
}

TYPED_TEST_P( MatrixTest, findElementTest )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_FindElement< MatrixType >();
}

TYPED_TEST_P( MatrixTest, forElements )
{
   using MatrixType = typename TestFixture::MatrixType;

   // SortedSegments have huge memory requirements for building traversing kernels so we just skip them
   if constexpr( ! TNL::Algorithms::Segments::isSortedSegments_v< typename MatrixType::SegmentsType > )
      test_ForElements< MatrixType >();
}

TYPED_TEST_P( MatrixTest, forElementsIf )
{
   using MatrixType = typename TestFixture::MatrixType;

   // SortedSegments have huge memory requirements for building traversing kernels so we just skip them
   if constexpr( ! TNL::Algorithms::Segments::isSortedSegments_v< typename MatrixType::SegmentsType > )
      test_ForElementsIf< MatrixType >();
}

TYPED_TEST_P( MatrixTest, forElementsWithArray )
{
   using MatrixType = typename TestFixture::MatrixType;

   // SortedSegments have huge memory requirements for building traversing kernels so we just skip them
   if constexpr( ! TNL::Algorithms::Segments::isSortedSegments_v< typename MatrixType::SegmentsType > )
      test_ForElementsWithArray< MatrixType >();
}

TYPED_TEST_P( MatrixTest, forRows )
{
   using MatrixType = typename TestFixture::MatrixType;

   // SortedSegments have huge memory requirements for building traversing kernels so we just skip them
   if constexpr( ! TNL::Algorithms::Segments::isSortedSegments_v< typename MatrixType::SegmentsType > )
      test_ForRows< MatrixType >();
}

TYPED_TEST_P( MatrixTest, reduceRows )
{
   using MatrixType = typename TestFixture::MatrixType;

   // SortedSegments have huge memory requirements for building reduction kernels so we just skip them
   if constexpr( ! TNL::Algorithms::Segments::isSortedSegments_v< typename MatrixType::SegmentsType > )
      test_reduceRows< MatrixType >();
}

TYPED_TEST_P( MatrixTest, sortColumnIndexes )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SortColumnIndexes< MatrixType >();
}

TYPED_TEST_P( MatrixTest, saveAndLoad )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_SaveAndLoad< MatrixType >( saveAndLoadFileName );
}

TYPED_TEST_P( MatrixTest, getTransposition )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_getTransposition< MatrixType >();
}

TYPED_TEST_P( MatrixTest, comparison )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_Comparison< MatrixType >();
}

REGISTER_TYPED_TEST_SUITE_P(
   MatrixTest,
   Constructors,
   setDimensionsTest,
   setRowCapacitiesTest,
   setLikeTest,
   setElementsTest,
   setElementsFromVectorTest,
   resetTest,
   getRowTest,
   setElementTest,
   addElementTest,
   findElementTest,
   forElements,
   forElementsIf,
   forElementsWithArray,
   forRows,
   reduceRows,
   sortColumnIndexes,
   saveAndLoad,
   getTransposition,
   comparison );
