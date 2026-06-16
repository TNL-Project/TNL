// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Containers/Vector.h>
#include <TNL/Graphs/Algorithms/maximalIndependentSet.h>
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
using MISVector =
   TNL::Containers::Vector< typename GraphType::IndexType, typename GraphType::DeviceType, typename GraphType::IndexType >;

template< typename GraphType >
using VertexIndexVector = MISVector< GraphType >;

template< typename GraphType >
GraphType
makeUndirectedGraph(
   typename GraphType::IndexType vertexCount,
   std::initializer_list<
      std::tuple< typename GraphType::IndexType, typename GraphType::IndexType, typename GraphType::ValueType > > edges )
{
   return GraphType( vertexCount, edges, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
}

template< typename GraphType >
void
expectComputedMISIsValid( const GraphType& graph )
{
   MISVector< GraphType > independentSet;
   TNL::Graphs::Algorithms::maximalIndependentSet( graph, independentSet );

   EXPECT_EQ( independentSet.getSize(), graph.getVertexCount() );
   if( graph.getVertexCount() > 0 )
      EXPECT_GT( TNL::sum( independentSet ), 0 );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSet ) );
}

template< typename GraphType, typename VertexIndexes >
void
expectComputedMISIsValid( const GraphType& graph, const VertexIndexes& vertexIndexes )
{
   MISVector< GraphType > independentSet;
   TNL::Graphs::Algorithms::maximalIndependentSet( graph, vertexIndexes, independentSet );

   EXPECT_EQ( independentSet.getSize(), graph.getVertexCount() );
   if( vertexIndexes.getSize() > 0 )
      EXPECT_GT( TNL::sum( independentSet ), 0 );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, vertexIndexes, independentSet ) );
}

template< typename GraphType, typename VertexPredicate >
void
expectComputedMISIsValidIf(
   const GraphType& graph,
   VertexPredicate&& vertexPredicate,
   typename GraphType::IndexType activeVerticesCount )
{
   MISVector< GraphType > independentSet;
   TNL::Graphs::Algorithms::maximalIndependentSetIf( graph, vertexPredicate, independentSet );

   EXPECT_EQ( independentSet.getSize(), graph.getVertexCount() );
   if( activeVerticesCount > 0 )
      EXPECT_GT( TNL::sum( independentSet ), 0 );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSetIf( graph, vertexPredicate, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_empty )
{
   using GraphType = typename TestFixture::GraphType;
   using MISVectorType = MISVector< GraphType >;

   GraphType graph;
   MISVectorType independentSet;

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_star_true )
{
   using GraphType = typename TestFixture::GraphType;
   using MISVectorType = MISVector< GraphType >;

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
   const MISVectorType independentSet( { 0, 1, 1, 1, 1 } );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_star_false )
{
   using GraphType = typename TestFixture::GraphType;
   using MISVectorType = MISVector< GraphType >;

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
   const MISVectorType independentSet( { 1, 1, 0, 0, 0 } );

   EXPECT_FALSE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_chain_true )
{
   using GraphType = typename TestFixture::GraphType;
   using MISVectorType = MISVector< GraphType >;

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
   const MISVectorType independentSet( { 1, 0, 1, 0, 1 } );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_chain_false )
{
   using GraphType = typename TestFixture::GraphType;
   using MISVectorType = MISVector< GraphType >;

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
   const MISVectorType independentSet( { 0, 0, 1, 0, 1 } );

   EXPECT_FALSE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_isolated_vertex_false )
{
   using GraphType = typename TestFixture::GraphType;
   using MISVectorType = MISVector< GraphType >;

   const GraphType graph = makeUndirectedGraph< GraphType >( 1, {} );
   const MISVectorType independentSet( { 0 } );

   EXPECT_FALSE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_withIndexes_true )
{
   using GraphType = typename TestFixture::GraphType;
   using MISVectorType = MISVector< GraphType >;
   using VertexIndexVectorType = VertexIndexVector< GraphType >;

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
   const VertexIndexVectorType vertexIndexes( { 1, 2, 3 } );
   const MISVectorType independentSet( { 0, 1, 0, 1, 0 } );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, vertexIndexes, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_withIndexes_false )
{
   using GraphType = typename TestFixture::GraphType;
   using MISVectorType = MISVector< GraphType >;
   using VertexIndexVectorType = VertexIndexVector< GraphType >;

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
   const VertexIndexVectorType vertexIndexes( { 1, 2, 3 } );
   const MISVectorType independentSet( { 0, 0, 0, 0, 0 } );

   EXPECT_FALSE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, vertexIndexes, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_withPredicate_true )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using MISVectorType = MISVector< GraphType >;

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
   const MISVectorType independentSet( { 0, 1, 0, 1, 0 } );
   const auto middleVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex >= 1 && vertex <= 3;
   };

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSetIf( graph, middleVertices, independentSet ) );
}

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_withPredicate_false )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using MISVectorType = MISVector< GraphType >;

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
   const MISVectorType independentSet( { 0, 0, 0, 0, 0 } );
   const auto middleVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex >= 1 && vertex <= 3;
   };

   EXPECT_FALSE( TNL::Graphs::Algorithms::isMaximalIndependentSetIf( graph, middleVertices, independentSet ) );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_empty )
{
   using GraphType = typename TestFixture::GraphType;
   using MISVectorType = MISVector< GraphType >;

   GraphType graph;
   MISVectorType independentSet;

   TNL::Graphs::Algorithms::maximalIndependentSet( graph, independentSet );

   EXPECT_EQ( independentSet.getSize(), 0 );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSet ) );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_single_vertex )
{
   using GraphType = typename TestFixture::GraphType;

   const GraphType graph = makeUndirectedGraph< GraphType >( 1, {} );

   expectComputedMISIsValid( graph );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_star )
{
   using GraphType = typename TestFixture::GraphType;

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

   expectComputedMISIsValid( graph );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_chain )
{
   using GraphType = typename TestFixture::GraphType;

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

   expectComputedMISIsValid( graph );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_withIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using VertexIndexVectorType = VertexIndexVector< GraphType >;

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
   const VertexIndexVectorType vertexIndexes( { 1, 2, 3 } );

   expectComputedMISIsValid( graph, vertexIndexes );
}

TYPED_TEST( GraphTest, test_maximalIndependentSetIf )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;

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
   const auto middleVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex >= 1 && vertex <= 3;
   };

   expectComputedMISIsValidIf( graph, middleVertices, static_cast< IndexType >( 3 ) );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_small )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 0, 3, 1.0 },
         { 1, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 3, 1.0 },
         { 2, 4, 1.0 },
      } );
   // clang-format on

   expectComputedMISIsValid( graph );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_medium )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      8,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 4, 1.0 },
         { 1, 3, 1.0 }, { 1, 5, 1.0 }, { 1, 6, 1.0 }, { 1, 7, 1.0 },
         { 2, 3, 1.0 }, { 2, 7, 1.0 }, { 3, 4, 1.0 }, { 4, 5, 1.0 },
         { 4, 6, 1.0 },
      } );
   // clang-format on

   expectComputedMISIsValid( graph );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_large )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      15,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 4, 1.0 }, { 0, 5, 1.0 }, { 0, 7, 1.0 },
         { 1, 3, 1.0 }, { 1, 8, 1.0 }, { 1, 12, 1.0 }, { 2, 5, 1.0 }, { 2, 10, 1.0 },
         { 2, 13, 1.0 }, { 3, 5, 1.0 }, { 3, 12, 1.0 }, { 3, 13, 1.0 }, { 4, 5, 1.0 },
         { 4, 6, 1.0 }, { 4, 7, 1.0 }, { 4, 11, 1.0 }, { 4, 13, 1.0 }, { 5, 11, 1.0 },
         { 5, 12, 1.0 }, { 5, 14, 1.0 }, { 6, 8, 1.0 }, { 6, 9, 1.0 }, { 6, 10, 1.0 },
         { 6, 12, 1.0 }, { 6, 14, 1.0 }, { 7, 9, 1.0 }, { 8, 10, 1.0 }, { 8, 14, 1.0 },
         { 10, 14, 1.0 }, { 13, 14, 1.0 },
      } );
   // clang-format on

   expectComputedMISIsValid( graph );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_edge_predicate_block_one_edge )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using MISVectorType = MISVector< GraphType >;

   // clang-format off
   // Triangle: 0-1, 1-2, 0-2 with different weights.
   const GraphType graph = makeUndirectedGraph< GraphType >(
      3,
      {
         { 0, 1, 1.0 },
         { 1, 2, 2.0 },
         { 0, 2, 1.0 },
      } );
   // clang-format on

   // Block edge with weight >= 2.0 (the edge 1-2).
   // In the filtered graph, 0 is adjacent to both 1 and 2, but 1 and 2 are not adjacent.
   // So {1, 2} should be a valid MIS.
   MISVectorType independentSet;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < 2.0;
   };

   TNL::Graphs::Algorithms::maximalIndependentSet( graph, edgePredicate, independentSet );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, edgePredicate, independentSet ) );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_edge_predicate_block_all )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using MISVectorType = MISVector< GraphType >;

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

   // Block all edges -> every vertex is isolated, so the entire vertex set is an MIS.
   MISVectorType independentSet;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType )
   {
      return false;
   };

   TNL::Graphs::Algorithms::maximalIndependentSet( graph, edgePredicate, independentSet );

   EXPECT_EQ( TNL::sum( independentSet ), graph.getVertexCount() );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, edgePredicate, independentSet ) );
}

TYPED_TEST( GraphTest, test_maximalIndependentSet_edge_predicate_identity )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using MISVectorType = MISVector< GraphType >;

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

   // Allow all edges -> same as whole-graph MIS.
   MISVectorType independentSet;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType )
   {
      return true;
   };

   TNL::Graphs::Algorithms::maximalIndependentSet( graph, edgePredicate, independentSet );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSet ) );
}

#include "../../main.h"
