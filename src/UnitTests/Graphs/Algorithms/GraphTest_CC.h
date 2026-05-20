#include <TNL/Graphs/Algorithms/connectedComponents.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <gtest/gtest.h>

template< typename Matrix >
class GraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::UndirectedGraph >;
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

template< typename GraphType >
GraphType
makeUndirectedGraph(
   typename GraphType::IndexType vertexCount,
   std::initializer_list<
      std::tuple< typename GraphType::IndexType, typename GraphType::IndexType, typename GraphType::ValueType > > edges )
{
   return GraphType( vertexCount, edges, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
}

TYPED_TEST( GraphTest, test_CC_empty )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph;
   ComponentsType components;

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   EXPECT_EQ( components.getSize(), 0 );
}

TYPED_TEST( GraphTest, test_CC_star )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 0, 2, 1.0 },
         { 0, 3, 1.0 },
         { 0, 4, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 5 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 0, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_chain )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 3, 4, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 5 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 0, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_chain_alternating )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      6,
      {
         { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 4, 1.0 },
         { 3, 5, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 6 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 1, 0, 1, 0, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_small )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      10,
      {
         { 2, 6, 1.0 },
         { 4, 6, 1.0 },
         { 4, 7, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 10 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 1, 2, 3, 2, 5, 2, 2, 8, 9 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_medium )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      15,
      {
         {  0,  4, 1.0 },
         {  0, 10, 1.0 },
         {  0, 13, 1.0 },
         {  2,  5, 1.0 },
         {  2,  6, 1.0 },
         {  2,  8, 1.0 },
         {  3, 10, 1.0 },
         {  3, 13, 1.0 },
         {  4, 12, 1.0 },
         {  5,  7, 1.0 },
         {  8,  9, 1.0 },
         { 12, 14, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 15 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 0, 11, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_medium2 )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      15,
      {
         {  0,  3, 1.0 }, {  0,  4, 1.0 }, {  0, 10, 1.0 }, {  2,  4, 1.0 }, {  2,  6, 1.0 },
         {  3,  4, 1.0 }, {  3,  5, 1.0 }, {  3,  9, 1.0 }, {  3, 13, 1.0 }, {  4, 11, 1.0 },
         {  5, 11, 1.0 }, {  5, 14, 1.0 }, {  8, 12, 1.0 }, {  9, 14, 1.0 }, { 10, 12, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 15 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 1, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_large )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      30,
      {
         {  1,  3, 1.0 }, {  1,  8, 1.0 }, {  2, 10, 1.0 }, {  3,  6, 1.0 }, {  3,  8, 1.0 }, {  5, 21, 1.0 },
         {  6, 11, 1.0 }, {  6, 14, 1.0 }, {  6, 22, 1.0 }, {  6, 23, 1.0 }, {  6, 25, 1.0 }, {  8, 11, 1.0 },
         {  8, 25, 1.0 }, {  8, 28, 1.0 }, { 10, 25, 1.0 }, { 11, 28, 1.0 }, { 12, 20, 1.0 }, { 14, 17, 1.0 },
         { 15, 24, 1.0 }, { 16, 29, 1.0 }, { 17, 21, 1.0 }, { 18, 19, 1.0 }, { 19, 26, 1.0 }, { 20, 24, 1.0 },
         { 21, 28, 1.0 }, { 24, 29, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 30 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected(
      { 0, 1, 1, 1, 4, 1, 1, 7, 1, 9, 1, 1, 12, 13, 1, 12, 12, 1, 18, 18, 12, 1, 1, 1, 12, 1, 18, 27, 1, 12 } );
   ASSERT_EQ( components, expected );
}

#include "../../main.h"
