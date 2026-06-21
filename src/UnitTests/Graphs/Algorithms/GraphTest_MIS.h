// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

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
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int, TNL::Matrices::SymmetricMatrix >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::SymmetricMatrix >
#elif defined( __CUDACC__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::SymmetricMatrix >
#elif defined( __HIP__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::SymmetricMatrix >
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

template< typename GraphType >
void
test_isMaximalIndependentSet_withPredicate_true_impl()
{
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

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_withPredicate_true )
{
   test_isMaximalIndependentSet_withPredicate_true_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_isMaximalIndependentSet_withPredicate_false_impl()
{
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

TYPED_TEST( GraphTest, test_isMaximalIndependentSet_withPredicate_false )
{
   test_isMaximalIndependentSet_withPredicate_false_impl< typename TestFixture::GraphType >();
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

template< typename GraphType >
void
test_maximalIndependentSetIf_impl()
{
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

TYPED_TEST( GraphTest, test_maximalIndependentSetIf )
{
   test_maximalIndependentSetIf_impl< typename TestFixture::GraphType >();
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

template< typename GraphType >
void
test_maximalIndependentSet_edge_predicate_block_one_edge_impl()
{
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

TYPED_TEST( GraphTest, test_maximalIndependentSet_edge_predicate_block_one_edge )
{
   test_maximalIndependentSet_edge_predicate_block_one_edge_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_maximalIndependentSet_edge_predicate_block_all_impl()
{
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

TYPED_TEST( GraphTest, test_maximalIndependentSet_edge_predicate_block_all )
{
   test_maximalIndependentSet_edge_predicate_block_all_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_maximalIndependentSet_edge_predicate_identity_impl()
{
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

TYPED_TEST( GraphTest, test_maximalIndependentSet_edge_predicate_identity )
{
   test_maximalIndependentSet_edge_predicate_identity_impl< typename TestFixture::GraphType >();
}

// clang-format off
// 10 vertices, same topology as graph A for BFS/SSSP/CC.
// Edges with weight 2 are "expensive".
//
//     0---1---2
//     |   |   |
//     3---4---5
//     |   |   |
//     6---7---8---9
//
// Weight-2 edges: 1-4, 4-5.  All others have weight 1.
// clang-format on

template< typename GraphType >
GraphType
makeMISGraphA()
{
   using Real = typename GraphType::ValueType;
   // clang-format off
   // 10 vertices, same topology as undirected graph A for CC.
   return GraphType(
      10,
      {
         { 0, 1, Real( 1 ) }, { 0, 3, Real( 1 ) },
         { 1, 2, Real( 1 ) }, { 1, 4, Real( 2 ) },
         { 2, 5, Real( 1 ) },
         { 3, 4, Real( 1 ) }, { 3, 6, Real( 1 ) },
         { 4, 5, Real( 2 ) }, { 4, 7, Real( 1 ) },
         { 5, 8, Real( 1 ) },
         { 6, 7, Real( 1 ) },
         { 7, 8, Real( 1 ) },
         { 8, 9, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeMISSubgraphB()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,3,4,6,7,9} -> remapped to {0,1,2,3,4,5,6}
   // clang-format off
   return GraphType(
      7,
      {
         { 0, 1, Real( 1 ) }, { 0, 2, Real( 1 ) },
         { 1, 3, Real( 2 ) },
         { 2, 3, Real( 1 ) }, { 2, 4, Real( 1 ) },
         { 3, 5, Real( 1 ) },
         { 4, 5, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeMISSubgraphD()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,2,3,5,6,7,8,9} -> remapped to {0,1,2,3,4,5,6,7,8}
   // clang-format off
   return GraphType(
      9,
      {
         { 0, 1, Real( 1 ) }, { 0, 3, Real( 1 ) },
         { 1, 2, Real( 1 ) },
         { 2, 4, Real( 1 ) },
         { 3, 5, Real( 1 ) },
         { 4, 7, Real( 1 ) },
         { 5, 6, Real( 1 ) },
         { 6, 7, Real( 1 ) },
         { 7, 8, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeMISSubgraphC()
{
   using Real = typename GraphType::ValueType;
   // All 10 vertices, edges with weight >= 2 removed.
   // clang-format off
   return GraphType(
      10,
      {
         { 0, 1, Real( 1 ) }, { 0, 3, Real( 1 ) },
         { 1, 2, Real( 1 ) },
         { 2, 5, Real( 1 ) },
         { 3, 4, Real( 1 ) }, { 3, 6, Real( 1 ) },
         { 4, 7, Real( 1 ) },
         { 5, 8, Real( 1 ) },
         { 6, 7, Real( 1 ) },
         { 7, 8, Real( 1 ) },
         { 8, 9, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeMISSubgraphE2()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,3,4,6,7}, edges with weight >= 2 also removed.
   // Remap: 0->0, 1->1, 3->2, 4->3, 6->4, 7->5
   // clang-format off
   return GraphType(
      6,
      {
         { 0, 1, Real( 1 ) }, { 0, 2, Real( 1 ) },
         { 2, 3, Real( 1 ) }, { 2, 4, Real( 1 ) },
         { 3, 5, Real( 1 ) },
         { 4, 5, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
void
test_MIS_subgraph_vertex_removal_predicate_impl()
{
   using IndexType = typename GraphType::IndexType;
   using MISVectorType = MISVector< GraphType >;

   const auto graphA = makeMISGraphA< GraphType >();
   const auto subgraphB = makeMISSubgraphB< GraphType >();

   const auto excludeVertices = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 2 && v != 5 && v != 8;
   };

   MISVectorType misA, misB;
   TNL::Graphs::Algorithms::maximalIndependentSetIf( graphA, excludeVertices, misA );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphB, misB );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSetIf( graphA, excludeVertices, misA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphB, misB ) );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   int sumA = 0, sumB = 0;
   for( int i = 0; i < (int) newToOld.size(); i++ ) {
      sumA += misA.getElement( newToOld[ i ] );
      sumB += misB.getElement( i );
   }
   EXPECT_EQ( sumA, sumB );
}

TYPED_TEST( GraphTest, test_MIS_subgraph_vertex_removal_predicate )
{
   test_MIS_subgraph_vertex_removal_predicate_impl< typename TestFixture::GraphType >();
}

TYPED_TEST( GraphTest, test_MIS_subgraph_vertex_removal_indexed )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using MISVectorType = MISVector< GraphType >;

   const auto graphA = makeMISGraphA< GraphType >();
   const auto subgraphB = makeMISSubgraphB< GraphType >();

   const MISVectorType vertexIndexes( { 0, 1, 3, 4, 6, 7, 9 } );

   MISVectorType misA, misB;
   TNL::Graphs::Algorithms::maximalIndependentSet( graphA, vertexIndexes, misA );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphB, misB );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graphA, vertexIndexes, misA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphB, misB ) );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   int sumA = 0, sumB = 0;
   for( int i = 0; i < (int) newToOld.size(); i++ ) {
      sumA += misA.getElement( newToOld[ i ] );
      sumB += misB.getElement( i );
   }
   EXPECT_EQ( sumA, sumB );
}

template< typename GraphType >
void
test_MIS_subgraph_vertex_removal_disconnected_impl()
{
   using IndexType = typename GraphType::IndexType;
   using MISVectorType = MISVector< GraphType >;

   const auto graphA = makeMISGraphA< GraphType >();
   const auto subgraphD = makeMISSubgraphD< GraphType >();

   const auto excludeFour = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 4;
   };

   MISVectorType misA, misD;
   TNL::Graphs::Algorithms::maximalIndependentSetIf( graphA, excludeFour, misA );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphD, misD );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSetIf( graphA, excludeFour, misA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphD, misD ) );

   const std::vector< int > newToOld = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
   int sumA = 0, sumD = 0;
   for( int i = 0; i < (int) newToOld.size(); i++ ) {
      sumA += misA.getElement( newToOld[ i ] );
      sumD += misD.getElement( i );
   }
   EXPECT_EQ( sumA, sumD );
}

TYPED_TEST( GraphTest, test_MIS_subgraph_vertex_removal_disconnected )
{
   test_MIS_subgraph_vertex_removal_disconnected_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_MIS_subgraph_edge_removal_wholeGraph_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using MISVectorType = MISVector< GraphType >;

   const auto graphA = makeMISGraphA< GraphType >();
   const auto subgraphC = makeMISSubgraphC< GraphType >();

   const auto blockWeight2 = [ = ] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < ValueType( 2 );
   };

   MISVectorType misA, misC;
   TNL::Graphs::Algorithms::maximalIndependentSet( graphA, blockWeight2, misA );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphC, misC );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graphA, blockWeight2, misA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphC, misC ) );

   EXPECT_EQ( TNL::sum( misA ), TNL::sum( misC ) );
}

TYPED_TEST( GraphTest, test_MIS_subgraph_edge_removal_wholeGraph )
{
   test_MIS_subgraph_edge_removal_wholeGraph_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_MIS_subgraph_edge_removal_withIndexes_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using MISVectorType = MISVector< GraphType >;

   const auto graphA = makeMISGraphA< GraphType >();
   const auto subgraphE2 = makeMISSubgraphE2< GraphType >();

   const MISVectorType vertexIndexes( { 0, 1, 3, 4, 6, 7 } );
   const auto blockWeight2 = [ = ] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < ValueType( 2 );
   };

   MISVectorType misA, misE2;
   TNL::Graphs::Algorithms::maximalIndependentSet( graphA, vertexIndexes, blockWeight2, misA );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphE2, misE2 );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graphA, vertexIndexes, blockWeight2, misA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphE2, misE2 ) );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7 };
   int sumA = 0, sumE2 = 0;
   for( int i = 0; i < (int) newToOld.size(); i++ ) {
      sumA += misA.getElement( newToOld[ i ] );
      sumE2 += misE2.getElement( i );
   }
   EXPECT_EQ( sumA, sumE2 );
}

TYPED_TEST( GraphTest, test_MIS_subgraph_edge_removal_withIndexes )
{
   test_MIS_subgraph_edge_removal_withIndexes_impl< typename TestFixture::GraphType >();
}

#include "../../main.h"
