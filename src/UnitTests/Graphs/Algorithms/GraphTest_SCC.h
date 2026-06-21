#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Graphs/Algorithms/stronglyConnectedComponents.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <gtest/gtest.h>

template< typename Matrix >
class GraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::DirectedGraph >;
};

using GraphTestTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >
#elif defined( __CUDACC__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int >
#endif
   >;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

TYPED_TEST( GraphTest, test_SCC_empty )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph;
   ComponentsType components;

   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );

   EXPECT_EQ( components.getSize(), 0 );
}

TYPED_TEST( GraphTest, test_SCC_small )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      10,
      {
         { 1, 5, 1.0 },
         { 2, 0, 1.0 }, { 2, 3, 1.0 }, { 2, 5, 1.0 },
         { 3, 0, 1.0 }, { 3, 1, 1.0 }, { 3, 5, 1.0 }, { 3, 7, 1.0 },
         { 5, 2, 1.0 }, { 5, 7, 1.0 },
         { 6, 1, 1.0 },
         { 7, 6, 1.0 },
         { 8, 3, 1.0 },
         { 9, 6, 1.0 }, { 9, 8, 1.0 },
      } );
   // clang-format on

   ComponentsType components( graph.getVertexCount(), 0 );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );

   // Expected SCC IDs: {0}=5, {1,2,3,5,6,7}=3, {4}=4, {8}=2, {9}=1
   ComponentsType expected( { 5, 3, 3, 3, 4, 3, 3, 3, 2, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_small2 )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      10,
      {
         { 0, 5, 1.0 },
         { 1, 3, 1.0 }, { 1, 6, 1.0 }, { 1, 9, 1.0 },
         { 2, 4, 1.0 },
         { 3, 1, 1.0 }, { 3, 9, 1.0 },
         { 4, 1, 1.0 },
         { 5, 2, 1.0 }, { 5, 4, 1.0 }, { 5, 7, 1.0 },
         { 6, 9, 1.0 },
         { 7, 1, 1.0 }, { 7, 2, 1.0 }, { 7, 5, 1.0 },
         { 8, 6, 1.0 },
      } );
   // clang-format on

   ComponentsType components( graph.getVertexCount(), 0 );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );

   ComponentsType expected( { 8, 6, 7, 6, 5, 3, 4, 3, 2, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_medium )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      15,
      {
         {  2,  7, 1.0 }, {  2,  8, 1.0 }, {  3,  0, 1.0 }, {  3,  2, 1.0 }, {  4, 11, 1.0 },
         {  6,  5, 1.0 }, {  6, 10, 1.0 }, {  7,  0, 1.0 }, {  7,  8, 1.0 }, {  8, 14, 1.0 },
         {  9, 11, 1.0 }, { 10,  4, 1.0 }, { 11, 10, 1.0 }, { 12,  4, 1.0 }, { 12, 10, 1.0 },
         { 12, 13, 1.0 }, { 13,  0, 1.0 }, { 13,  4, 1.0 }, { 13,  5, 1.0 }, { 14,  9, 1.0 },
      } );
   // clang-format on

   ComponentsType components( graph.getVertexCount(), 0 );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );

   ComponentsType expected( { 13, 12, 11, 10, 4, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_medium2 )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      15,
      {
         {  0,  1, 1.0 }, {  0,  7, 1.0 }, {  1, 11, 1.0 }, {  2,  6, 1.0 }, {  2,  7, 1.0 },
         {  3,  7, 1.0 }, {  3,  8, 1.0 }, {  3,  9, 1.0 }, {  4,  6, 1.0 }, {  4,  7, 1.0 },
         {  4, 14, 1.0 }, {  5,  4, 1.0 }, {  5,  7, 1.0 }, {  5, 12, 1.0 }, {  6,  1, 1.0 },
         {  6,  9, 1.0 }, {  6, 12, 1.0 }, {  7,  5, 1.0 }, {  7,  8, 1.0 }, {  7, 13, 1.0 },
         {  8,  2, 1.0 }, {  8, 10, 1.0 }, {  8, 11, 1.0 }, {  8, 14, 1.0 }, {  9, 10, 1.0 },
         { 10,  0, 1.0 }, { 10,  5, 1.0 }, { 10,  7, 1.0 }, { 10, 11, 1.0 }, { 11,  1, 1.0 },
         { 11,  5, 1.0 }, { 11,  9, 1.0 }, { 11, 12, 1.0 }, { 11, 13, 1.0 }, { 12,  0, 1.0 },
         { 12,  5, 1.0 }, { 12,  8, 1.0 }, { 12, 10, 1.0 }, { 13,  1, 1.0 }, { 13,  6, 1.0 },
         { 13, 10, 1.0 }, { 13, 14, 1.0 }, { 14,  4, 1.0 }, { 14,  9, 1.0 },
      } );
   // clang-format on

   ComponentsType components( graph.getVertexCount(), 0 );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );

   ComponentsType expected( { 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_large )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      30,
      {
         {  0,  9, 1.0 }, {  0, 21, 1.0 }, {  0, 24, 1.0 }, {  1,  5, 1.0 }, {  1,  6, 1.0 }, {  1,  7, 1.0 },
         {  1,  8, 1.0 }, {  1, 29, 1.0 }, {  2,  1, 1.0 }, {  2,  3, 1.0 }, {  2,  7, 1.0 }, {  2, 12, 1.0 },
         {  2, 26, 1.0 }, {  2, 27, 1.0 }, {  2, 29, 1.0 }, {  3,  4, 1.0 }, {  3, 16, 1.0 }, {  3, 28, 1.0 },
         {  4,  0, 1.0 }, {  4, 13, 1.0 }, {  4, 15, 1.0 }, {  4, 18, 1.0 }, {  4, 25, 1.0 }, {  4, 26, 1.0 },
         {  5,  1, 1.0 }, {  5,  2, 1.0 }, {  5,  4, 1.0 }, {  5,  6, 1.0 }, {  5, 18, 1.0 }, {  5, 27, 1.0 },
         {  5, 28, 1.0 }, {  6,  9, 1.0 }, {  6, 10, 1.0 }, {  7,  0, 1.0 }, {  7,  2, 1.0 }, {  7, 11, 1.0 },
         {  7, 12, 1.0 }, {  7, 21, 1.0 }, {  7, 28, 1.0 }, {  7, 29, 1.0 }, {  8,  7, 1.0 }, {  8, 17, 1.0 },
         {  8, 28, 1.0 }, {  9,  0, 1.0 }, {  9,  3, 1.0 }, {  9,  6, 1.0 }, {  9, 22, 1.0 }, { 10, 15, 1.0 },
         { 10, 28, 1.0 }, { 10, 29, 1.0 }, { 11,  7, 1.0 }, { 11, 12, 1.0 }, { 11, 13, 1.0 }, { 11, 22, 1.0 },
         { 12,  2, 1.0 }, { 12, 10, 1.0 }, { 12, 11, 1.0 }, { 12, 24, 1.0 }, { 12, 25, 1.0 }, { 13,  3, 1.0 },
         { 13, 14, 1.0 }, { 13, 15, 1.0 }, { 13, 21, 1.0 }, { 13, 27, 1.0 }, { 14,  1, 1.0 }, { 14, 18, 1.0 },
         { 14, 26, 1.0 }, { 15,  7, 1.0 }, { 15,  8, 1.0 }, { 15, 20, 1.0 }, { 15, 29, 1.0 }, { 16,  6, 1.0 },
         { 16, 15, 1.0 }, { 16, 17, 1.0 }, { 16, 25, 1.0 }, { 17,  2, 1.0 }, { 17,  4, 1.0 }, { 17,  6, 1.0 },
         { 17, 24, 1.0 }, { 18,  5, 1.0 }, { 18,  8, 1.0 }, { 18,  9, 1.0 }, { 18, 16, 1.0 }, { 18, 22, 1.0 },
         { 18, 25, 1.0 }, { 19,  2, 1.0 }, { 19, 12, 1.0 }, { 19, 15, 1.0 }, { 19, 25, 1.0 }, { 20,  0, 1.0 },
         { 20, 13, 1.0 }, { 20, 15, 1.0 }, { 20, 16, 1.0 }, { 21,  3, 1.0 }, { 21,  9, 1.0 }, { 21, 11, 1.0 },
         { 21, 18, 1.0 }, { 22, 15, 1.0 }, { 22, 18, 1.0 }, { 23,  3, 1.0 }, { 23, 17, 1.0 }, { 24,  5, 1.0 },
         { 24, 22, 1.0 }, { 25,  9, 1.0 }, { 25, 20, 1.0 }, { 25, 29, 1.0 }, { 26,  4, 1.0 }, { 26,  5, 1.0 },
         { 26, 10, 1.0 }, { 26, 12, 1.0 }, { 27,  9, 1.0 }, { 27, 10, 1.0 }, { 27, 13, 1.0 }, { 27, 25, 1.0 },
         { 28,  5, 1.0 }, { 28,  6, 1.0 }, { 28,  7, 1.0 }, { 28, 12, 1.0 }, { 28, 13, 1.0 }, { 28, 17, 1.0 },
         { 28, 19, 1.0 }, { 29,  0, 1.0 },
      } );
   // clang-format on

   ComponentsType components( graph.getVertexCount(), 0 );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );

   ComponentsType expected( { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_large2 )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      30,
      {
         {  0, 23, 1.0 }, {  1,  5, 1.0 }, {  1, 15, 1.0 }, {  2, 20, 1.0 }, {  3,  8, 1.0 }, {  3, 13, 1.0 },
         {  3, 16, 1.0 }, {  4,  3, 1.0 }, {  4, 27, 1.0 }, {  6,  0, 1.0 }, {  7,  9, 1.0 }, {  8,  4, 1.0 },
         {  8, 27, 1.0 }, { 10,  1, 1.0 }, { 10, 16, 1.0 }, { 10, 18, 1.0 }, { 11,  5, 1.0 }, { 11, 17, 1.0 },
         { 14,  2, 1.0 }, { 17, 29, 1.0 }, { 18,  6, 1.0 }, { 18,  7, 1.0 }, { 18, 20, 1.0 }, { 19,  0, 1.0 },
         { 20,  7, 1.0 }, { 21,  9, 1.0 }, { 21, 16, 1.0 }, { 23, 26, 1.0 }, { 25,  2, 1.0 }, { 26,  4, 1.0 },
         { 27, 12, 1.0 }, { 28,  3, 1.0 }, { 28, 15, 1.0 }, { 29,  1, 1.0 }, { 29,  5, 1.0 },
      } );
   // clang-format on

   ComponentsType components( graph.getVertexCount(), 0 );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );

   ComponentsType expected(
      { 28, 27, 26, 22, 22, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_huge_sparse )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      100,
      {
         {  1, 36, 1.0 }, {  3, 99, 1.0 }, {  4, 38, 1.0 }, {  4, 42, 1.0 }, {  5, 22, 1.0 }, {  5, 94, 1.0 },
         {  6,  1, 1.0 }, {  6, 77, 1.0 }, {  6, 97, 1.0 }, {  7, 79, 1.0 }, {  7, 80, 1.0 }, {  8, 50, 1.0 },
         { 10, 80, 1.0 }, { 10, 86, 1.0 }, { 13, 34, 1.0 }, { 13, 44, 1.0 }, { 13, 66, 1.0 }, { 15, 90, 1.0 },
         { 16, 23, 1.0 }, { 16, 25, 1.0 }, { 16, 48, 1.0 }, { 18, 44, 1.0 }, { 18, 75, 1.0 }, { 19, 82, 1.0 },
         { 19, 97, 1.0 }, { 21, 27, 1.0 }, { 21, 42, 1.0 }, { 22, 82, 1.0 }, { 23,  4, 1.0 }, { 23, 68, 1.0 },
         { 24, 46, 1.0 }, { 26, 71, 1.0 }, { 26, 85, 1.0 }, { 27, 59, 1.0 }, { 27, 92, 1.0 }, { 28, 11, 1.0 },
         { 28, 98, 1.0 }, { 30, 70, 1.0 }, { 31,  0, 1.0 }, { 32, 58, 1.0 }, { 32, 85, 1.0 }, { 35,  6, 1.0 },
         { 36, 27, 1.0 }, { 36, 34, 1.0 }, { 37, 11, 1.0 }, { 39, 79, 1.0 }, { 39, 89, 1.0 }, { 42, 39, 1.0 },
         { 45, 18, 1.0 }, { 48, 76, 1.0 }, { 49, 15, 1.0 }, { 50, 81, 1.0 }, { 50, 84, 1.0 }, { 50, 96, 1.0 },
         { 51, 21, 1.0 }, { 51, 24, 1.0 }, { 51, 69, 1.0 }, { 51, 79, 1.0 }, { 53, 74, 1.0 }, { 56, 81, 1.0 },
         { 57, 38, 1.0 }, { 58, 85, 1.0 }, { 58, 99, 1.0 }, { 59, 79, 1.0 }, { 61, 49, 1.0 }, { 61, 58, 1.0 },
         { 62, 78, 1.0 }, { 63, 88, 1.0 }, { 65, 21, 1.0 }, { 65, 95, 1.0 }, { 66, 36, 1.0 }, { 66, 71, 1.0 },
         { 67, 13, 1.0 }, { 68, 66, 1.0 }, { 69, 63, 1.0 }, { 70, 90, 1.0 }, { 71, 20, 1.0 }, { 71, 21, 1.0 },
         { 72, 78, 1.0 }, { 73, 27, 1.0 }, { 73, 33, 1.0 }, { 75, 64, 1.0 }, { 78, 15, 1.0 }, { 78, 45, 1.0 },
         { 79,  1, 1.0 }, { 81, 39, 1.0 }, { 83, 16, 1.0 }, { 83, 33, 1.0 }, { 83, 69, 1.0 }, { 85,  7, 1.0 },
         { 85, 54, 1.0 }, { 85, 79, 1.0 }, { 85, 83, 1.0 }, { 90, 81, 1.0 }, { 93,  3, 1.0 }, { 93, 30, 1.0 },
         { 94, 81, 1.0 }, { 95, 35, 1.0 }, { 97, 78, 1.0 }, { 97, 79, 1.0 }, { 98, 60, 1.0 },
      } );
   // clang-format on

   ComponentsType components( graph.getVertexCount(), 0 );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );

   ComponentsType expected( { 96, 21, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78,
                              77, 76, 75, 74, 73, 72, 71, 21, 70, 69, 68, 67, 66, 65, 64, 63, 21, 62, 61, 60,
                              59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 21,
                              40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
                              20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_huge_dense )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      100,
      {
         {  0, 24, 1.0 }, {  0, 54, 1.0 }, {  1, 14, 1.0 }, {  1, 24, 1.0 }, {  1, 33, 1.0 }, {  1, 52, 1.0 },
         {  2, 29, 1.0 }, {  2, 45, 1.0 }, {  2, 57, 1.0 }, {  3, 46, 1.0 }, {  3, 50, 1.0 }, {  3, 62, 1.0 },
         {  3, 82, 1.0 }, {  4,  0, 1.0 }, {  4, 31, 1.0 }, {  4, 56, 1.0 }, {  4, 69, 1.0 }, {  4, 74, 1.0 },
         {  4, 79, 1.0 }, {  5,  2, 1.0 }, {  5, 21, 1.0 }, {  5, 22, 1.0 }, {  5, 37, 1.0 }, {  5, 74, 1.0 },
         {  5, 78, 1.0 }, {  5, 89, 1.0 }, {  6, 12, 1.0 }, {  6, 49, 1.0 }, {  6, 55, 1.0 }, {  6, 57, 1.0 },
         {  6, 84, 1.0 }, {  7,  2, 1.0 }, {  7, 63, 1.0 }, {  7, 79, 1.0 }, {  7, 87, 1.0 }, {  8,  1, 1.0 },
         {  8, 44, 1.0 }, {  8, 56, 1.0 }, {  8, 80, 1.0 }, {  9, 34, 1.0 }, {  9, 54, 1.0 }, {  9, 56, 1.0 },
         {  9, 73, 1.0 }, {  9, 90, 1.0 }, {  9, 98, 1.0 }, { 10,  6, 1.0 }, { 10, 14, 1.0 }, { 10, 22, 1.0 },
         { 10, 27, 1.0 }, { 10, 48, 1.0 }, { 10, 57, 1.0 }, { 11, 14, 1.0 }, { 11, 35, 1.0 }, { 11, 56, 1.0 },
         { 11, 61, 1.0 }, { 11, 62, 1.0 }, { 11, 87, 1.0 }, { 11, 99, 1.0 }, { 12,  0, 1.0 }, { 12,  3, 1.0 },
         { 12,  9, 1.0 }, { 12, 16, 1.0 }, { 12, 30, 1.0 }, { 12, 40, 1.0 }, { 12, 41, 1.0 }, { 12, 68, 1.0 },
         { 12, 84, 1.0 }, { 12, 89, 1.0 }, { 13, 35, 1.0 }, { 13, 83, 1.0 }, { 13, 96, 1.0 }, { 14,  6, 1.0 },
         { 14, 28, 1.0 }, { 14, 33, 1.0 }, { 14, 44, 1.0 }, { 14, 72, 1.0 }, { 15, 36, 1.0 }, { 15, 39, 1.0 },
         { 15, 66, 1.0 }, { 15, 95, 1.0 }, { 16,  2, 1.0 }, { 16, 20, 1.0 }, { 16, 58, 1.0 }, { 16, 82, 1.0 },
         { 16, 97, 1.0 }, { 17, 19, 1.0 }, { 17, 21, 1.0 }, { 17, 50, 1.0 }, { 17, 55, 1.0 }, { 18, 15, 1.0 },
         { 18, 35, 1.0 }, { 18, 41, 1.0 }, { 18, 58, 1.0 }, { 18, 82, 1.0 }, { 18, 89, 1.0 }, { 19, 15, 1.0 },
         { 19, 22, 1.0 }, { 19, 30, 1.0 }, { 19, 31, 1.0 }, { 19, 36, 1.0 }, { 19, 70, 1.0 }, { 19, 87, 1.0 },
         { 20,  4, 1.0 }, { 20, 23, 1.0 }, { 20, 40, 1.0 }, { 20, 41, 1.0 }, { 20, 47, 1.0 }, { 20, 55, 1.0 },
         { 20, 65, 1.0 }, { 20, 67, 1.0 }, { 20, 80, 1.0 }, { 21, 70, 1.0 }, { 21, 74, 1.0 }, { 21, 83, 1.0 },
         { 21, 88, 1.0 }, { 21, 98, 1.0 }, { 21, 99, 1.0 }, { 22,  3, 1.0 }, { 22, 13, 1.0 }, { 22, 32, 1.0 },
         { 22, 38, 1.0 }, { 22, 49, 1.0 }, { 22, 56, 1.0 }, { 22, 76, 1.0 }, { 23, 42, 1.0 }, { 23, 65, 1.0 },
         { 23, 73, 1.0 }, { 23, 82, 1.0 }, { 23, 84, 1.0 }, { 24,  4, 1.0 }, { 24, 26, 1.0 }, { 24, 48, 1.0 },
         { 24, 70, 1.0 }, { 24, 95, 1.0 }, { 25, 44, 1.0 }, { 25, 52, 1.0 }, { 25, 71, 1.0 }, { 25, 81, 1.0 },
         { 25, 97, 1.0 }, { 26, 24, 1.0 }, { 26, 35, 1.0 }, { 26, 63, 1.0 }, { 26, 84, 1.0 }, { 26, 87, 1.0 },
         { 27,  4, 1.0 }, { 27,  5, 1.0 }, { 27, 24, 1.0 }, { 27, 47, 1.0 }, { 27, 56, 1.0 }, { 28,  4, 1.0 },
         { 28,  7, 1.0 }, { 28, 14, 1.0 }, { 28, 30, 1.0 }, { 28, 34, 1.0 }, { 28, 45, 1.0 }, { 28, 67, 1.0 },
         { 28, 82, 1.0 }, { 29, 32, 1.0 }, { 30, 16, 1.0 }, { 30, 17, 1.0 }, { 30, 32, 1.0 }, { 30, 47, 1.0 },
         { 31, 21, 1.0 }, { 31, 26, 1.0 }, { 31, 38, 1.0 }, { 31, 41, 1.0 }, { 31, 57, 1.0 }, { 31, 78, 1.0 },
         { 31, 84, 1.0 }, { 32,  5, 1.0 }, { 32, 16, 1.0 }, { 32, 21, 1.0 }, { 32, 45, 1.0 }, { 32, 61, 1.0 },
         { 32, 69, 1.0 }, { 32, 95, 1.0 }, { 33, 34, 1.0 }, { 34, 21, 1.0 }, { 34, 48, 1.0 }, { 34, 65, 1.0 },
         { 34, 67, 1.0 }, { 35,  4, 1.0 }, { 35, 31, 1.0 }, { 35, 39, 1.0 }, { 35, 40, 1.0 }, { 35, 44, 1.0 },
         { 35, 51, 1.0 }, { 35, 75, 1.0 }, { 35, 85, 1.0 }, { 35, 97, 1.0 }, { 36, 10, 1.0 }, { 36, 46, 1.0 },
         { 36, 47, 1.0 }, { 36, 82, 1.0 }, { 37, 10, 1.0 }, { 37, 39, 1.0 }, { 37, 60, 1.0 }, { 37, 84, 1.0 },
         { 37, 89, 1.0 }, { 37, 90, 1.0 }, { 38,  2, 1.0 }, { 38, 30, 1.0 }, { 38, 42, 1.0 }, { 38, 50, 1.0 },
         { 38, 51, 1.0 }, { 38, 68, 1.0 }, { 39, 36, 1.0 }, { 39, 48, 1.0 }, { 39, 68, 1.0 }, { 40, 55, 1.0 },
         { 40, 57, 1.0 }, { 40, 65, 1.0 }, { 40, 72, 1.0 }, { 40, 73, 1.0 }, { 40, 86, 1.0 }, { 41, 12, 1.0 },
         { 41, 46, 1.0 }, { 41, 88, 1.0 }, { 42,  5, 1.0 }, { 42, 38, 1.0 }, { 43,  1, 1.0 }, { 43,  7, 1.0 },
         { 43,  8, 1.0 }, { 43,  9, 1.0 }, { 43, 12, 1.0 }, { 43, 13, 1.0 }, { 43, 19, 1.0 }, { 43, 20, 1.0 },
         { 43, 30, 1.0 }, { 43, 47, 1.0 }, { 43, 56, 1.0 }, { 43, 82, 1.0 }, { 43, 91, 1.0 }, { 43, 94, 1.0 },
         { 43, 95, 1.0 }, { 44,  2, 1.0 }, { 44, 38, 1.0 }, { 44, 65, 1.0 }, { 44, 87, 1.0 }, { 45,  7, 1.0 },
         { 45, 29, 1.0 }, { 45, 46, 1.0 }, { 46, 34, 1.0 }, { 46, 63, 1.0 }, { 46, 74, 1.0 }, { 46, 88, 1.0 },
         { 46, 97, 1.0 }, { 47,  5, 1.0 }, { 47, 14, 1.0 }, { 47, 24, 1.0 }, { 47, 58, 1.0 }, { 48, 22, 1.0 },
         { 48, 73, 1.0 }, { 48, 87, 1.0 }, { 48, 90, 1.0 }, { 49, 25, 1.0 }, { 49, 43, 1.0 }, { 49, 51, 1.0 },
         { 49, 65, 1.0 }, { 49, 77, 1.0 }, { 49, 82, 1.0 }, { 49, 94, 1.0 }, { 50, 22, 1.0 }, { 50, 87, 1.0 },
         { 51,  1, 1.0 }, { 51, 16, 1.0 }, { 51, 17, 1.0 }, { 51, 22, 1.0 }, { 51, 27, 1.0 }, { 51, 31, 1.0 },
         { 51, 52, 1.0 }, { 51, 68, 1.0 }, { 51, 74, 1.0 }, { 52,  5, 1.0 }, { 52, 68, 1.0 }, { 52, 99, 1.0 },
         { 53, 14, 1.0 }, { 53, 20, 1.0 }, { 53, 36, 1.0 }, { 53, 42, 1.0 }, { 54,  7, 1.0 }, { 54,  8, 1.0 },
         { 54, 25, 1.0 }, { 54, 34, 1.0 }, { 54, 66, 1.0 }, { 54, 79, 1.0 }, { 55, 29, 1.0 }, { 55, 32, 1.0 },
         { 55, 43, 1.0 }, { 55, 77, 1.0 }, { 55, 91, 1.0 }, { 55, 97, 1.0 }, { 56,  1, 1.0 }, { 56, 32, 1.0 },
         { 56, 50, 1.0 }, { 57,  0, 1.0 }, { 57, 99, 1.0 }, { 58, 26, 1.0 }, { 58, 66, 1.0 }, { 58, 88, 1.0 },
         { 59,  7, 1.0 }, { 59, 27, 1.0 }, { 59, 51, 1.0 }, { 59, 76, 1.0 }, { 59, 91, 1.0 }, { 60, 20, 1.0 },
         { 60, 43, 1.0 }, { 60, 46, 1.0 }, { 60, 70, 1.0 }, { 60, 97, 1.0 }, { 61, 58, 1.0 }, { 61, 64, 1.0 },
         { 61, 72, 1.0 }, { 61, 81, 1.0 }, { 61, 92, 1.0 }, { 62,  1, 1.0 }, { 62, 13, 1.0 }, { 62, 22, 1.0 },
         { 62, 34, 1.0 }, { 62, 47, 1.0 }, { 62, 59, 1.0 }, { 62, 71, 1.0 }, { 62, 81, 1.0 }, { 62, 96, 1.0 },
         { 63,  4, 1.0 }, { 63,  8, 1.0 }, { 63, 12, 1.0 }, { 63, 23, 1.0 }, { 63, 67, 1.0 }, { 63, 71, 1.0 },
         { 63, 72, 1.0 }, { 63, 74, 1.0 }, { 63, 90, 1.0 }, { 64,  2, 1.0 }, { 64, 38, 1.0 }, { 64, 48, 1.0 },
         { 64, 49, 1.0 }, { 64, 60, 1.0 }, { 64, 77, 1.0 }, { 64, 84, 1.0 }, { 65, 26, 1.0 }, { 65, 42, 1.0 },
         { 65, 49, 1.0 }, { 65, 98, 1.0 }, { 66,  0, 1.0 }, { 66,  7, 1.0 }, { 66, 17, 1.0 }, { 66, 24, 1.0 },
         { 66, 38, 1.0 }, { 66, 81, 1.0 }, { 66, 85, 1.0 }, { 67, 28, 1.0 }, { 67, 52, 1.0 }, { 67, 61, 1.0 },
         { 67, 74, 1.0 }, { 67, 80, 1.0 }, { 68,  6, 1.0 }, { 68, 14, 1.0 }, { 68, 24, 1.0 }, { 68, 44, 1.0 },
         { 68, 50, 1.0 }, { 68, 73, 1.0 }, { 68, 84, 1.0 }, { 68, 97, 1.0 }, { 69,  0, 1.0 }, { 69, 17, 1.0 },
         { 69, 29, 1.0 }, { 69, 30, 1.0 }, { 69, 43, 1.0 }, { 69, 52, 1.0 }, { 69, 95, 1.0 }, { 70, 11, 1.0 },
         { 70, 16, 1.0 }, { 70, 29, 1.0 }, { 70, 75, 1.0 }, { 70, 91, 1.0 }, { 71,  6, 1.0 }, { 71, 12, 1.0 },
         { 71, 88, 1.0 }, { 72, 24, 1.0 }, { 72, 25, 1.0 }, { 72, 33, 1.0 }, { 72, 34, 1.0 }, { 72, 58, 1.0 },
         { 72, 64, 1.0 }, { 72, 99, 1.0 }, { 73,  1, 1.0 }, { 73,  6, 1.0 }, { 73, 17, 1.0 }, { 73, 19, 1.0 },
         { 73, 22, 1.0 }, { 73, 68, 1.0 }, { 73, 97, 1.0 }, { 74,  8, 1.0 }, { 74, 30, 1.0 }, { 74, 38, 1.0 },
         { 74, 42, 1.0 }, { 74, 47, 1.0 }, { 74, 82, 1.0 }, { 74, 97, 1.0 }, { 75,  2, 1.0 }, { 75, 16, 1.0 },
         { 75, 18, 1.0 }, { 75, 46, 1.0 }, { 75, 88, 1.0 }, { 76, 16, 1.0 }, { 76, 22, 1.0 }, { 77, 76, 1.0 },
         { 77, 81, 1.0 }, { 78,  5, 1.0 }, { 78,  9, 1.0 }, { 78, 27, 1.0 }, { 78, 28, 1.0 }, { 78, 54, 1.0 },
         { 78, 79, 1.0 }, { 78, 96, 1.0 }, { 79, 33, 1.0 }, { 79, 57, 1.0 }, { 80, 26, 1.0 }, { 80, 58, 1.0 },
         { 81, 21, 1.0 }, { 81, 59, 1.0 }, { 81, 95, 1.0 }, { 81, 96, 1.0 }, { 82, 12, 1.0 }, { 82, 29, 1.0 },
         { 82, 36, 1.0 }, { 82, 52, 1.0 }, { 82, 54, 1.0 }, { 82, 63, 1.0 }, { 83, 32, 1.0 }, { 83, 48, 1.0 },
         { 83, 58, 1.0 }, { 83, 65, 1.0 }, { 84,  2, 1.0 }, { 84, 14, 1.0 }, { 84, 15, 1.0 }, { 84, 58, 1.0 },
         { 85, 21, 1.0 }, { 85, 53, 1.0 }, { 85, 57, 1.0 }, { 85, 67, 1.0 }, { 85, 69, 1.0 }, { 86, 23, 1.0 },
         { 86, 61, 1.0 }, { 86, 77, 1.0 }, { 87, 11, 1.0 }, { 87, 51, 1.0 }, { 87, 68, 1.0 }, { 88, 17, 1.0 },
         { 88, 34, 1.0 }, { 88, 89, 1.0 }, { 88, 97, 1.0 }, { 89,  7, 1.0 }, { 89, 21, 1.0 }, { 89, 49, 1.0 },
         { 90,  7, 1.0 }, { 90, 31, 1.0 }, { 90, 45, 1.0 }, { 90, 55, 1.0 }, { 90, 62, 1.0 }, { 90, 64, 1.0 },
         { 90, 78, 1.0 }, { 90, 81, 1.0 }, { 91,  6, 1.0 }, { 91, 19, 1.0 }, { 91, 20, 1.0 }, { 91, 29, 1.0 },
         { 91, 49, 1.0 }, { 91, 75, 1.0 }, { 91, 97, 1.0 }, { 92, 29, 1.0 }, { 92, 35, 1.0 }, { 92, 40, 1.0 },
         { 92, 48, 1.0 }, { 93,  2, 1.0 }, { 93,  9, 1.0 }, { 93, 22, 1.0 }, { 93, 29, 1.0 }, { 93, 57, 1.0 },
         { 93, 71, 1.0 }, { 93, 77, 1.0 }, { 93, 81, 1.0 }, { 93, 98, 1.0 }, { 94, 15, 1.0 }, { 94, 18, 1.0 },
         { 94, 62, 1.0 }, { 95, 37, 1.0 }, { 95, 47, 1.0 }, { 95, 58, 1.0 }, { 95, 66, 1.0 }, { 95, 70, 1.0 },
         { 95, 73, 1.0 }, { 96,  1, 1.0 }, { 96, 33, 1.0 }, { 96, 56, 1.0 }, { 96, 60, 1.0 }, { 96, 80, 1.0 },
         { 96, 95, 1.0 }, { 97, 18, 1.0 }, { 97, 25, 1.0 }, { 97, 57, 1.0 }, { 97, 61, 1.0 }, { 97, 74, 1.0 },
         { 97, 99, 1.0 }, { 98,  6, 1.0 }, { 98, 27, 1.0 }, { 98, 29, 1.0 }, { 98, 85, 1.0 }, { 99, 26, 1.0 },
         { 99, 39, 1.0 }, { 99, 75, 1.0 }, { 99, 77, 1.0 }, { 99, 95, 1.0 },
      } );
   // clang-format on

   ComponentsType components( graph.getVertexCount(), 0 );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );

   ComponentsType expected( { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_indexed_small )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      10,
      {
         { 1, 5, 1.0 },
         { 2, 0, 1.0 }, { 2, 3, 1.0 }, { 2, 5, 1.0 },
         { 3, 0, 1.0 }, { 3, 1, 1.0 }, { 3, 5, 1.0 }, { 3, 7, 1.0 },
         { 5, 2, 1.0 }, { 5, 7, 1.0 },
         { 6, 1, 1.0 },
         { 7, 6, 1.0 },
         { 8, 3, 1.0 },
         { 9, 6, 1.0 }, { 9, 8, 1.0 },
      } );
   // clang-format on

   // Select only vertices {1, 3, 5, 6, 7} which form a subgraph.
   // In the full graph: SCCs are {0,2,3,5,1,7,6}, {8}, {9}, {4}.
   // Restricting to {1,3,5,6,7}: edges among them include 1->5, 3->1, 3->5, 3->7, 5->7, 6->1, 7->6.
   // The cycle 1->5->7->6->1 makes {1,5,6,7} one SCC; 3 reaches into it but is not reached back -> own SCC.
   ComponentsType vertexIndexes( { 1, 3, 5, 6, 7 } );
   ComponentsType components;

   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, vertexIndexes, components );

   // Vertex 0,2,4,8,9 are inactive -> -1
   // Active: 1,3,5,6,7 -- 1,5,6,7 form SCC (label 1), 3 forms singleton (label 2)
   ComponentsType expected( { -1, 1, -1, 2, -1, 1, 1, 1, -1, -1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_indexed_all_vertices )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      10,
      {
         { 1, 5, 1.0 },
         { 2, 0, 1.0 }, { 2, 3, 1.0 }, { 2, 5, 1.0 },
         { 3, 0, 1.0 }, { 3, 1, 1.0 }, { 3, 5, 1.0 }, { 3, 7, 1.0 },
         { 5, 2, 1.0 }, { 5, 7, 1.0 },
         { 6, 1, 1.0 },
         { 7, 6, 1.0 },
         { 8, 3, 1.0 },
         { 9, 6, 1.0 }, { 9, 8, 1.0 },
      } );
   // clang-format on

   // All vertices active -> same as whole-graph SCC.
   ComponentsType vertexIndexes( { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 } );
   ComponentsType components;

   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, vertexIndexes, components );

   ComponentsType expected( { 5, 3, 3, 3, 4, 3, 3, 3, 2, 1 } );
   ASSERT_EQ( components, expected );
}

template< typename GraphType >
void
test_SCC_predicate_none_active_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 1.0 },
         { 2, 0, 1.0 },
      } );
   // clang-format on

   // No vertex is active -> all should get -1.
   ComponentsType components;
   auto predicate = [] __cuda_callable__( IndexType )
   {
      return false;
   };

   TNL::Graphs::Algorithms::stronglyConnectedComponentsIf( graph, predicate, components );

   ComponentsType expected( { -1, -1, -1, -1, -1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_predicate_none_active )
{
   test_SCC_predicate_none_active_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_SCC_predicate_select_subset_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      6,
      {
         { 0, 1, 1.0 },
         { 1, 2, 1.0 },
         { 2, 0, 1.0 },
         { 3, 4, 1.0 },
         { 4, 5, 1.0 },
         { 5, 3, 1.0 },
      } );
   // clang-format on

   // Full graph: two SCCs {0,1,2} and {3,4,5}.
   // Predicate: select only vertices < 3 -> {0,1,2} form one SCC.
   ComponentsType components;
   auto predicate = [] __cuda_callable__( IndexType v )
   {
      return v < 3;
   };

   TNL::Graphs::Algorithms::stronglyConnectedComponentsIf( graph, predicate, components );

   ComponentsType expected( { 1, 1, 1, -1, -1, -1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_predicate_select_subset )
{
   test_SCC_predicate_select_subset_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_SCC_edge_predicate_break_cycle_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   // Cycle: 0 -> 1 -> 2 -> 0 (all weight 1.0), plus a side edge 1 -> 2 with weight 99.0
   // Actually make it simple: one cycle 0->1(w=1), 1->2(w=1), 2->0(w=2)
   const GraphType graph(
      3,
      {
         { 0, 1, 1.0 },
         { 1, 2, 1.0 },
         { 2, 0, 2.0 },
      } );
   // clang-format on

   // Block edge with weight >= 2.0 (the back-edge 2->0).
   // Without it: 0 can reach 1,2 but not return from 2 -> each vertex is its own SCC.
   ComponentsType components;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < 2.0;
   };

   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, edgePredicate, components );

   ComponentsType expected( { 3, 2, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_edge_predicate_break_cycle )
{
   test_SCC_edge_predicate_break_cycle_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_SCC_edge_predicate_identity_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      10,
      {
         { 1, 5, 1.0 },
         { 2, 0, 1.0 }, { 2, 3, 1.0 }, { 2, 5, 1.0 },
         { 3, 0, 1.0 }, { 3, 1, 1.0 }, { 3, 5, 1.0 }, { 3, 7, 1.0 },
         { 5, 2, 1.0 }, { 5, 7, 1.0 },
         { 6, 1, 1.0 },
         { 7, 6, 1.0 },
         { 8, 3, 1.0 },
         { 9, 6, 1.0 }, { 9, 8, 1.0 },
      } );
   // clang-format on

   // Allow all edges -> same as whole-graph SCC.
   ComponentsType components;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType )
   {
      return true;
   };

   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, edgePredicate, components );

   ComponentsType expected( { 5, 3, 3, 3, 4, 3, 3, 3, 2, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_edge_predicate_identity )
{
   test_SCC_edge_predicate_identity_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_SCC_vertex_and_edge_predicate_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   // Two cycles: 0->1->2->0 and 3->4->5->3
   const GraphType graph(
      6,
      {
         { 0, 1, 1.0 },
         { 1, 2, 2.0 },
         { 2, 0, 1.0 },
         { 3, 4, 1.0 },
         { 4, 5, 1.0 },
         { 5, 3, 1.0 },
      } );
   // clang-format on

   // Vertex predicate: only vertices 0,1,2 active.
   // Edge predicate: block weight >= 2.0 (blocks 1->2).
   // With the edge blocked, in the induced subgraph {0,1,2}:
   // 0 reaches 1 (via 0->1), but 1 cannot reach 0 (1->2 is blocked).
   // So all three are singletons.
   ComponentsType components;
   auto vertexPredicate = [] __cuda_callable__( IndexType v )
   {
      return v < 3;
   };
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < 2.0;
   };

   TNL::Graphs::Algorithms::stronglyConnectedComponentsIf( graph, vertexPredicate, edgePredicate, components );

   ComponentsType expected( { 3, 2, 1, -1, -1, -1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_SCC_vertex_and_edge_predicate )
{
   test_SCC_vertex_and_edge_predicate_impl< typename TestFixture::GraphType >();
}

template< typename VectorA, typename VectorB >
void
expectPartitionEquiv( const VectorA& compA, const VectorB& compB, const std::vector< int >& oldToNew, int origSize )
{
   for( int u = 0; u < origSize; u++ ) {
      if( oldToNew[ u ] < 0 )
         continue;
      for( int v = u + 1; v < origSize; v++ ) {
         if( oldToNew[ v ] < 0 )
            continue;
         bool sameCompA = compA.getElement( u ) == compA.getElement( v );
         bool sameCompB = compB.getElement( oldToNew[ u ] ) == compB.getElement( oldToNew[ v ] );
         ASSERT_EQ( sameCompA, sameCompB ) << "vertices " << u << " and " << v;
      }
   }
}

template< typename GraphType >
GraphType
makeDirectedSCCGraphA()
{
   // clang-format off
   return GraphType(
      10,
      {
         { 1, 5, 1 }, { 2, 0, 1 }, { 2, 3, 1 }, { 2, 5, 1 },
         { 3, 0, 1 }, { 3, 1, 1 }, { 3, 5, 1 }, { 3, 7, 1 },
         { 5, 2, 1 }, { 5, 7, 1 },
         { 6, 1, 1 }, { 7, 6, 1 },
         { 8, 3, 1 }, { 9, 6, 1 }, { 9, 8, 1 },
      } );
   // clang-format on
}

template< typename GraphType >
GraphType
makeSCCSubgraphB()
{
   // Vertices {1,2,3,5,6,7,8,9} -> remapped to {0,1,2,3,4,5,6,7}
   // Removed vertices {0,4}. Vertex 4 was isolated anyway.
   // clang-format off
   return GraphType(
      8,
      {
         { 0, 3, 1 }, { 1, 2, 1 }, { 1, 3, 1 },
         { 2, 0, 1 }, { 2, 3, 1 }, { 2, 5, 1 },
         { 3, 1, 1 }, { 3, 5, 1 },
         { 4, 0, 1 }, { 5, 4, 1 },
         { 6, 2, 1 }, { 7, 4, 1 }, { 7, 6, 1 },
      } );
   // clang-format on
}

template< typename GraphType >
GraphType
makeSCCSubgraphD()
{
   // Vertex {4} removed (was isolated). Remaining: {0,1,2,3,5,6,7,8,9}
   // Remap: 0->0, 1->1, 2->2, 3->3, 5->4, 6->5, 7->6, 8->7, 9->8
   // clang-format off
   return GraphType(
      9,
      {
         { 1, 4, 1 }, { 2, 0, 1 }, { 2, 3, 1 }, { 2, 4, 1 },
         { 3, 0, 1 }, { 3, 1, 1 }, { 3, 4, 1 }, { 3, 6, 1 },
         { 4, 2, 1 }, { 4, 6, 1 },
         { 5, 1, 1 }, { 6, 5, 1 },
         { 7, 3, 1 }, { 8, 5, 1 }, { 8, 7, 1 },
      } );
   // clang-format on
}

template< typename GraphType >
GraphType
makeSCCSubgraphC()
{
   // All 10 vertices, edge {5,2} removed (breaks the large SCC cycle).
   // clang-format off
   return GraphType(
      10,
      {
         { 1, 5, 1 }, { 2, 0, 1 }, { 2, 3, 1 }, { 2, 5, 1 },
         { 3, 0, 1 }, { 3, 1, 1 }, { 3, 5, 1 }, { 3, 7, 1 },
         { 5, 7, 1 },
         { 6, 1, 1 }, { 7, 6, 1 },
         { 8, 3, 1 }, { 9, 6, 1 }, { 9, 8, 1 },
      } );
   // clang-format on
}

template< typename GraphType >
GraphType
makeSCCSubgraphE2()
{
   // Vertices {1,2,3,5,6,7} with edge {5,2} also removed.
   // Remap: 1->0, 2->1, 3->2, 5->3, 6->4, 7->5
   // clang-format off
   return GraphType(
      6,
      {
         { 0, 3, 1 }, { 1, 2, 1 }, { 1, 3, 1 },
         { 2, 0, 1 }, { 2, 3, 1 }, { 2, 5, 1 },
         { 3, 5, 1 },
         { 4, 0, 1 }, { 5, 4, 1 },
      } );
   // clang-format on
}

template< typename GraphType >
void
test_SCC_subgraph_vertex_removal_predicate_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedSCCGraphA< GraphType >();
   const auto subgraphB = makeSCCSubgraphB< GraphType >();

   const auto exclude04 = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 0 && v != 4;
   };

   ComponentsType compA, compB;
   TNL::Graphs::Algorithms::stronglyConnectedComponentsIf( graphA, exclude04, compA );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( subgraphB, compB );

   const std::vector< int > oldToNew = { -1, 0, 1, 2, -1, 3, 4, 5, 6, 7 };
   expectPartitionEquiv( compA, compB, oldToNew, 10 );
}

TYPED_TEST( GraphTest, test_SCC_subgraph_vertex_removal_predicate )
{
   test_SCC_subgraph_vertex_removal_predicate_impl< typename TestFixture::GraphType >();
}

TYPED_TEST( GraphTest, test_SCC_subgraph_vertex_removal_indexed )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedSCCGraphA< GraphType >();
   const auto subgraphB = makeSCCSubgraphB< GraphType >();

   const ComponentsType vertexIndexes( { 1, 2, 3, 5, 6, 7, 8, 9 } );

   ComponentsType compA, compB;
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graphA, vertexIndexes, compA );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( subgraphB, compB );

   const std::vector< int > oldToNew = { -1, 0, 1, 2, -1, 3, 4, 5, 6, 7 };
   expectPartitionEquiv( compA, compB, oldToNew, 10 );
}

template< typename GraphType >
void
test_SCC_subgraph_vertex_removal_disconnected_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedSCCGraphA< GraphType >();
   const auto subgraphD = makeSCCSubgraphD< GraphType >();

   const auto excludeFour = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 4;
   };

   ComponentsType compA, compD;
   TNL::Graphs::Algorithms::stronglyConnectedComponentsIf( graphA, excludeFour, compA );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( subgraphD, compD );

   const std::vector< int > oldToNew = { 0, 1, 2, 3, -1, 4, 5, 6, 7, 8 };
   expectPartitionEquiv( compA, compD, oldToNew, 10 );
}

TYPED_TEST( GraphTest, test_SCC_subgraph_vertex_removal_disconnected )
{
   test_SCC_subgraph_vertex_removal_disconnected_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_SCC_subgraph_edge_removal_wholeGraph_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedSCCGraphA< GraphType >();
   const auto subgraphC = makeSCCSubgraphC< GraphType >();

   const auto blockEdge52 = [ = ] __cuda_callable__( IndexType src, IndexType tgt, ValueType )
   {
      return ! ( src == 5 && tgt == 2 );
   };

   ComponentsType compA, compC;
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graphA, blockEdge52, compA );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( subgraphC, compC );

   const std::vector< int > identity = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
   expectPartitionEquiv( compA, compC, identity, 10 );
}

TYPED_TEST( GraphTest, test_SCC_subgraph_edge_removal_wholeGraph )
{
   test_SCC_subgraph_edge_removal_wholeGraph_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_SCC_subgraph_edge_removal_withIndexes_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedSCCGraphA< GraphType >();
   const auto subgraphE2 = makeSCCSubgraphE2< GraphType >();

   const ComponentsType vertexIndexes( { 1, 2, 3, 5, 6, 7 } );
   const auto blockEdge52 = [ = ] __cuda_callable__( IndexType src, IndexType tgt, ValueType )
   {
      return ! ( src == 5 && tgt == 2 );
   };

   ComponentsType compA, compE2;
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graphA, vertexIndexes, blockEdge52, compA );
   TNL::Graphs::Algorithms::stronglyConnectedComponents( subgraphE2, compE2 );

   const std::vector< int > oldToNew = { -1, 0, 1, 2, -1, 3, 4, 5, -1, -1 };
   expectPartitionEquiv( compA, compE2, oldToNew, 10 );
}

TYPED_TEST( GraphTest, test_SCC_subgraph_edge_removal_withIndexes )
{
   test_SCC_subgraph_edge_removal_withIndexes_impl< typename TestFixture::GraphType >();
}

#include "../../main.h"
