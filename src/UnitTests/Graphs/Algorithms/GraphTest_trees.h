#include <iostream>
#include <cstdint>
#include <TNL/Graphs/Algorithms/trees.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Graphs/Graph.h>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class GraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::UndirectedGraph >;
};

// types for which MatrixTest is instantiated
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
GraphType
makeUndirectedGraph(
   typename GraphType::IndexType vertexCount,
   std::initializer_list<
      std::tuple< typename GraphType::IndexType, typename GraphType::IndexType, typename GraphType::ValueType > > edges )
{
   return GraphType( vertexCount, edges, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
}

// clang-format off
// Main tree topology (10 vertices, used across most tree tests).
// 9 edges, no cycles — a valid tree rooted at 0.
//
//            0
//           / \
//          1   2
//         / \ / \
//        3  4 5  6
//        |  | |
//        7  8 9
// clang-format on

TYPED_TEST( GraphTest, test_isTree_small )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 1 },
        { 1, 3, 1 }, { 1, 4, 1 },
        { 2, 5, 1 }, { 2, 6, 1 },
        { 3, 7, 1 },
        { 4, 8, 1 },
        { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
   ASSERT_TRUE( TNL::Graphs::Algorithms::isTree( graph ) );
}

TYPED_TEST( GraphTest, test_isTree_not_tree )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   // Same tree + extra edge (5,0) creates cycle 0-2-5-0.  Not a tree.
   //
   //            0
   //           / \
   //          1   2
   //         / \ / \
   //        3  4 5  6
   //        |  | |
   //        7  8 9
   // clang-format on
   GraphType graph(
      10,
      { { 0, 1, 1 },
        { 0, 2, 1 },
        { 1, 3, 1 },
        { 1, 4, 1 },
        { 2, 5, 1 },
        { 2, 6, 1 },
        { 3, 7, 1 },
        { 4, 8, 1 },
        { 5, 0, 1 },
        { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   ASSERT_FALSE( TNL::Graphs::Algorithms::isTree( graph ) );

   GraphType graph2(
      10,
      { { 0, 1, 1 }, { 0, 2, 1 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 0, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   ASSERT_FALSE( TNL::Graphs::Algorithms::isTree( graph2 ) );
}

TYPED_TEST( GraphTest, test_large_tree )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType tree(
      29,
      { { 3, 18, 1.0 },  { 0, 11, 1.0 },  { 4, 19, 1.0 },  { 2, 28, 1.0 },  { 1, 5, 1.0 },   { 8, 12, 1.0 },  { 8, 26, 1.0 },
        { 11, 28, 1.0 }, { 16, 23, 1.0 }, { 17, 23, 1.0 }, { 13, 19, 1.0 }, { 15, 25, 1.0 }, { 18, 25, 1.0 }, { 25, 20, 1.0 },
        { 3, 10, 2.0 },  { 4, 12, 2.0 },  { 1, 25, 2.0 },  { 7, 19, 2.0 },  { 10, 12, 2.0 }, { 10, 23, 2.0 }, { 14, 18, 2.0 },
        { 27, 28, 2.0 }, { 24, 28, 2.0 }, { 0, 22, 3.0 },  { 6, 11, 3.0 },  { 9, 17, 3.0 },  { 21, 23, 3.0 }, { 8, 27, 4.0 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   ASSERT_TRUE( TNL::Graphs::Algorithms::isTree( tree ) );
}

TYPED_TEST( GraphTest, test_small_forest )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   // Forest (5 vertices): tree {0,3,4}, isolated {1}, {2}.
   //
   //   0     1     2
   //  / \
   // 3   4
   // clang-format on
   GraphType graph( 5, { { 0, 3, 1.0 }, { 0, 4, 1.0 } }, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   ASSERT_FALSE( TNL::Graphs::Algorithms::isTree( graph ) );
   ASSERT_TRUE( TNL::Graphs::Algorithms::isForest( graph ) );

   IndexVector roots( { 0, 1, 2 } );
   ASSERT_TRUE( TNL::Graphs::Algorithms::isForestWithRoots( graph, roots ) );
}

// Subgraph tests for isTree

TYPED_TEST( GraphTest, test_isTree_subgraph_vertex_removal_predicate )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;

   // Main tree topology (see diagram above test_isTree_small).
   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 1 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   auto isActive = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 6 && v != 9;
   };

   ASSERT_TRUE( TNL::Graphs::Algorithms::isTreeIf( graph, 0, isActive ) );
}

TYPED_TEST( GraphTest, test_isTree_subgraph_vertex_removal_indexed )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 1 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   IndexVector vertexIndexes( { 0, 1, 2, 3, 4, 5, 7, 8 } );

   ASSERT_TRUE( TNL::Graphs::Algorithms::isTree( graph, 0, vertexIndexes ) );
}

TYPED_TEST( GraphTest, test_isTree_subgraph_edge_removal_wholeGraph )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;

   // Tree with weighted edges: weight 1 and weight 2
   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 2 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   // Block edge (0,2) -> disconnects the graph
   auto blockEdge02 = [ = ] __cuda_callable__( IndexType src, IndexType tgt, ValueType )
   {
      return ! ( ( src == 0 && tgt == 2 ) || ( src == 2 && tgt == 0 ) );
   };

   ASSERT_FALSE( TNL::Graphs::Algorithms::isTree( graph, 0, blockEdge02 ) );

   // Allow all edges -> is a tree
   auto allowAll = [] __cuda_callable__( IndexType, IndexType, ValueType )
   {
      return true;
   };

   ASSERT_TRUE( TNL::Graphs::Algorithms::isTree( graph, 0, allowAll ) );
}

TYPED_TEST( GraphTest, test_isTree_subgraph_edge_removal_withIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // Tree with weighted edges
   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 2 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   // Restrict to vertices {0,1,2,3,5} and block edge (0,2)
   // Induced subgraph without edge (0,2): {0,1,3} connected, {5} isolated -> not a tree
   IndexVector vertexIndexes( { 0, 1, 2, 3, 5 } );
   auto blockEdge02 = [ = ] __cuda_callable__( IndexType src, IndexType tgt, ValueType )
   {
      return ! ( ( src == 0 && tgt == 2 ) || ( src == 2 && tgt == 0 ) );
   };

   ASSERT_FALSE( TNL::Graphs::Algorithms::isTree( graph, 0, vertexIndexes, blockEdge02 ) );
}

// Subgraph tests for isForest

TYPED_TEST( GraphTest, test_isForest_subgraph_vertex_removal_predicate )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;

   // Tree on 10 vertices
   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 1 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   // Remove vertex 0 -> becomes a forest of 3 trees: {1,3,4,7,8}, {2,5,6,9}
   auto excludeZero = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 0;
   };

   ASSERT_FALSE( TNL::Graphs::Algorithms::isTreeIf( graph, 1, excludeZero ) );
   ASSERT_TRUE( TNL::Graphs::Algorithms::isForestIf( graph, excludeZero ) );
}

TYPED_TEST( GraphTest, test_isForest_subgraph_vertex_removal_indexed )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 1 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   // Remove vertex 0 -> forest
   IndexVector vertexIndexes( { 1, 2, 3, 4, 5, 6, 7, 8, 9 } );

   ASSERT_TRUE( TNL::Graphs::Algorithms::isForest( graph, vertexIndexes ) );
}

TYPED_TEST( GraphTest, test_isForest_subgraph_edge_removal_wholeGraph )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;

   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 2 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   // Block edge (0,2) -> graph becomes a forest (two trees: {0,1,3,4,7,8} and {2,5,6,9})
   auto blockEdge02 = [ = ] __cuda_callable__( IndexType src, IndexType tgt, ValueType )
   {
      return ! ( ( src == 0 && tgt == 2 ) || ( src == 2 && tgt == 0 ) );
   };

   ASSERT_TRUE( TNL::Graphs::Algorithms::isForest( graph, blockEdge02 ) );
}

TYPED_TEST( GraphTest, test_isForest_subgraph_edge_removal_withIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 2 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   // Vertices {0,1,2,3,5}, block edge (0,2)
   // Two components: {0,1,3} and {5} -> forest
   IndexVector vertexIndexes( { 0, 1, 2, 3, 5 } );
   auto blockEdge02 = [ = ] __cuda_callable__( IndexType src, IndexType tgt, ValueType )
   {
      return ! ( ( src == 0 && tgt == 2 ) || ( src == 2 && tgt == 0 ) );
   };

   ASSERT_TRUE( TNL::Graphs::Algorithms::isForest( graph, vertexIndexes, blockEdge02 ) );
}

// Subgraph tests for isForestWithRoots

TYPED_TEST( GraphTest, test_isForestWithRoots_basic )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph( 5, { { 0, 3, 1.0 }, { 0, 4, 1.0 } }, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   IndexVector roots( { 0, 1, 2 } );
   ASSERT_TRUE( TNL::Graphs::Algorithms::isForestWithRoots( graph, roots ) );
}

TYPED_TEST( GraphTest, test_isForestWithRoots_subgraph_indexed )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 1 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   // Remove vertex 0 -> forest with two trees rooted at 1 and 2
   IndexVector vertexIndexes( { 1, 2, 3, 4, 5, 6, 7, 8, 9 } );
   IndexVector roots( { 1, 2 } );

   ASSERT_TRUE( TNL::Graphs::Algorithms::isForestWithRoots( graph, vertexIndexes, roots ) );
}

TYPED_TEST( GraphTest, test_isForestWithRoots_subgraph_edge_removal_withIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph(
      10,
      { { 0, 1, 1 }, { 0, 2, 2 }, { 1, 3, 1 }, { 1, 4, 1 }, { 2, 5, 1 }, { 2, 6, 1 }, { 3, 7, 1 }, { 4, 8, 1 }, { 5, 9, 1 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );

   // Vertices {0,1,2,3,5}, block edge (0,2), roots {0,5}
   IndexVector vertexIndexes( { 0, 1, 2, 3, 5 } );
   auto blockEdge02 = [ = ] __cuda_callable__( IndexType src, IndexType tgt, ValueType )
   {
      return ! ( ( src == 0 && tgt == 2 ) || ( src == 2 && tgt == 0 ) );
   };
   IndexVector roots( { 0, 5 } );

   ASSERT_TRUE( TNL::Graphs::Algorithms::isForestWithRoots( graph, vertexIndexes, blockEdge02, roots ) );
}

#include "../../main.h"
