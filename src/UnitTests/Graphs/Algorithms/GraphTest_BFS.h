#pragma once

#include <TNL/Graphs/Algorithms/breadthFirstSearch.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class GraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::DirectedGraph >;
};

// types for which MatrixTest is instantiated
using GraphTestTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Sequential, int >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int >
#elif defined( __CUDACC__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int >
#endif
   >;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

TYPED_TEST( GraphTest, test_BFS_empty )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph;
   VectorType distances;
   TNL::Graphs::Algorithms::breadthFirstSearch( graph, 0, distances );
   EXPECT_EQ( distances.getSize(), 0 );
}

TYPED_TEST( GraphTest, test_BFS_small )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // Create a sample graph.
   // clang-format off
   GraphType graph(
        5, // number of nodes
        {  // edges definition
                         {0, 1, 1.0}, {0, 2, 1.0},
            {1, 0, 1.0},                           {1, 3, 1.0}, {1, 4, 1.0},
            {2, 0, 1.0},                           {2, 3, 1.0},
                         {3, 1, 1.0}, {3, 2, 1.0},              {3, 4, 1.0},
                         {4, 1, 1.0},              {4, 3, 1.0},
        });
   // clang-format on

   VectorType distances( graph.getVertexCount() );
   std::vector< VectorType > expectedDistances = {
      { 0, 1, 1, 2, 2 }, { 1, 0, 2, 1, 1 }, { 1, 2, 0, 1, 2 }, { 2, 1, 1, 0, 1 }, { 2, 1, 2, 1, 0 },
   };

   for( IndexType start_node = 0; start_node < graph.getVertexCount(); ++start_node ) {
      TNL::Graphs::Algorithms::breadthFirstSearch( graph, start_node, distances );
      ASSERT_EQ( distances, expectedDistances[ start_node ] ) << "start_node: " << start_node;
   }
}

TYPED_TEST( GraphTest, test_BFS_larger )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // Create a sample graph.
   // clang-format off
   GraphType graph(
        10, // number of graph nodes
        {   // edges definition
                         {0, 1, 1.0}, {0, 2, 1.0},
            {1, 0, 1.0},                           {1, 3, 1.0}, {1, 4, 1.0},
            {2, 0, 1.0}, {2, 3, 1.0},                                        {2, 5, 1.0},
                         {3, 1, 1.0}, {3, 2, 1.0},              {3, 4, 1.0},             {3, 6, 1.0},
                         {4, 1, 1.0},              {4, 3, 1.0},                                       {4, 7, 1.0},
                                      {5, 2, 1.0},                                       {5, 6, 1.0},               {5, 8, 1.0},
                                                   {6, 3, 1.0},              {6, 5, 1.0},                                        {6, 9, 1.0},
                                                                {7, 4, 1.0},                                        {7, 8, 1.0},
                                                                                     {8, 5, 1.0},              {8, 7, 1.0},              {8, 9, 1.0},
                                                                                                  {9, 6, 1.0},               {9, 8, 1.0},
        });
   // clang-format on

   VectorType distances( graph.getVertexCount() );
   std::vector< VectorType > expectedDistances = {
      { 0, 1, 1, 2, 2, 2, 3, 3, 3, 4 }, { 1, 0, 2, 1, 1, 3, 2, 2, 3, 3 }, { 1, 2, 0, 1, 2, 1, 2, 3, 2, 3 },
      { 2, 1, 1, 0, 1, 2, 1, 2, 3, 2 }, { 2, 1, 2, 1, 0, 3, 2, 1, 2, 3 }, { 2, 3, 1, 2, 3, 0, 1, 2, 1, 2 },
      { 3, 2, 2, 1, 2, 1, 0, 3, 2, 1 }, { 3, 2, 3, 2, 1, 2, 3, 0, 1, 2 }, { 3, 3, 2, 3, 2, 1, 2, 1, 0, 1 },
      { 4, 3, 3, 2, 3, 2, 1, 2, 1, 0 },
   };

   for( IndexType start_node = 0; start_node < graph.getVertexCount(); ++start_node ) {
      TNL::Graphs::Algorithms::breadthFirstSearch( graph, start_node, distances );
      ASSERT_EQ( distances, expectedDistances[ start_node ] ) << "start_node: " << start_node;
   }
}

TYPED_TEST( GraphTest, test_BFS_largest )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // Create a sample graph with 15 nodes.
   GraphType graph(
      15,  // number of graph nodes
      {
         // definition of edges
         { 0, 1, 1.0 },   { 0, 3, 1.0 },   { 1, 0, 1.0 },  { 1, 2, 1.0 },   { 1, 4, 1.0 },   { 2, 1, 1.0 },   { 2, 5, 1.0 },
         { 3, 0, 1.0 },   { 3, 4, 1.0 },   { 3, 6, 1.0 },  { 4, 1, 1.0 },   { 4, 3, 1.0 },   { 4, 5, 1.0 },   { 4, 7, 1.0 },
         { 5, 2, 1.0 },   { 5, 4, 1.0 },   { 5, 8, 1.0 },  { 6, 3, 1.0 },   { 6, 7, 1.0 },   { 6, 9, 1.0 },   { 7, 4, 1.0 },
         { 7, 6, 1.0 },   { 7, 8, 1.0 },   { 7, 10, 1.0 }, { 8, 5, 1.0 },   { 8, 7, 1.0 },   { 8, 11, 1.0 },  { 9, 6, 1.0 },
         { 9, 10, 1.0 },  { 9, 12, 1.0 },  { 10, 7, 1.0 }, { 10, 9, 1.0 },  { 10, 11, 1.0 }, { 10, 13, 1.0 }, { 11, 8, 1.0 },
         { 11, 10, 1.0 }, { 11, 14, 1.0 }, { 12, 9, 1.0 }, { 12, 13, 1.0 }, { 13, 10, 1.0 }, { 13, 12, 1.0 }, { 13, 14, 1.0 },
         { 14, 11, 1.0 }, { 14, 13, 1.0 },
      } );

   VectorType distances( graph.getVertexCount() );
   std::vector< VectorType > expectedDistances = {
      { 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6 }, { 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5 },
      { 2, 1, 0, 3, 2, 1, 4, 3, 2, 5, 4, 3, 6, 5, 4 }, { 1, 2, 3, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5 },
      { 2, 1, 2, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4 }, { 3, 2, 1, 2, 1, 0, 3, 2, 1, 4, 3, 2, 5, 4, 3 },
      { 2, 3, 4, 1, 2, 3, 0, 1, 2, 1, 2, 3, 2, 3, 4 }, { 3, 2, 3, 2, 1, 2, 1, 0, 1, 2, 1, 2, 3, 2, 3 },
      { 4, 3, 2, 3, 2, 1, 2, 1, 0, 3, 2, 1, 4, 3, 2 }, { 3, 4, 5, 2, 3, 4, 1, 2, 3, 0, 1, 2, 1, 2, 3 },
      { 4, 3, 4, 3, 2, 3, 2, 1, 2, 1, 0, 1, 2, 1, 2 }, { 5, 4, 3, 4, 3, 2, 3, 2, 1, 2, 1, 0, 3, 2, 1 },
      { 4, 5, 6, 3, 4, 5, 2, 3, 4, 1, 2, 3, 0, 1, 2 }, { 5, 4, 5, 4, 3, 4, 3, 2, 3, 2, 1, 2, 1, 0, 1 },
      { 6, 5, 4, 5, 4, 3, 4, 3, 2, 3, 2, 1, 2, 1, 0 }
   };

   for( IndexType start_node = 0; start_node < graph.getVertexCount(); start_node++ ) {
      TNL::Graphs::Algorithms::breadthFirstSearch( graph, start_node, distances );
      ASSERT_EQ( distances, expectedDistances[ start_node ] ) << "start_node: " << start_node;
   }
}

TYPED_TEST( GraphTest, test_BFS_withVertexIndexes_inducedSubgraph )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 }, { 0, 4, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 4, 3, 1.0 },
      } );
   // clang-format on
   const VectorType vertexIndexes( { 0, 1, 3 } );
   const VectorType expectedDistances( { 0, 1, -1, -1, -1 } );
   VectorType distances;

   TNL::Graphs::Algorithms::breadthFirstSearch( TNL::Graphs::makeSubGraph( graph, vertexIndexes ), 0, distances );

   ASSERT_EQ( distances, expectedDistances );
}

template< typename GraphType >
void
test_BFSIf_inducedSubgraph_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 }, { 0, 4, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 4, 3, 1.0 },
      } );
   // clang-format on
   const VectorType expectedDistances( { 0, 1, 2, -1, -1 } );
   VectorType distances;
   const auto firstThreeVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex <= 2;
   };

   TNL::Graphs::Algorithms::breadthFirstSearch( TNL::Graphs::makeSubGraph( graph, firstThreeVertices ), 0, distances );

   ASSERT_EQ( distances, expectedDistances );
}

TYPED_TEST( GraphTest, test_BFSIf_inducedSubgraph )
{
   test_BFSIf_inducedSubgraph_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_withVertexIndexes_visitor_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 }, { 0, 4, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 4, 3, 1.0 },
      } );
   // clang-format on
   const VectorType vertexIndexes( { 0, 1, 2 } );
   const VectorType expectedDistances( { 0, 1, 2, -1, -1 } );
   VectorType distances;
   VectorType visitedDistances( graph.getVertexCount(), -1 );
   auto visitedDistancesView = visitedDistances.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      visitedDistancesView[ vertex ] = distance;
   };

   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      TNL::Graphs::makeSubGraph( graph, vertexIndexes ), 0, visitor, distances );

   ASSERT_EQ( distances, expectedDistances );
   EXPECT_EQ( visitedDistances.getElement( 0 ), -1 );
   EXPECT_EQ( visitedDistances.getElement( 1 ), 1 );
   EXPECT_EQ( visitedDistances.getElement( 2 ), 2 );
   EXPECT_EQ( visitedDistances.getElement( 3 ), -1 );
   EXPECT_EQ( visitedDistances.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_BFS_withVertexIndexes_visitor )
{
   test_BFS_withVertexIndexes_visitor_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_byEdges_wholeGraph_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 }, { 0, 4, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 4, 3, 1.0 },
      } );
   // clang-format on

   const VectorType expectedDistances( { 0, 1, -1, 2, 1 } );
   VectorType distances;
   const auto forbidOneToTwo =
      [ = ] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType weight )
   {
      return ! ( source == 1 && target == 2 );
   };

   TNL::Graphs::Algorithms::breadthFirstSearch(
      TNL::Graphs::makeSubGraph( graph, TNL::Graphs::edgeOnly, forbidOneToTwo ), 0, distances );

   ASSERT_EQ( distances, expectedDistances );
}

TYPED_TEST( GraphTest, test_BFS_byEdges_wholeGraph )
{
   test_BFS_byEdges_wholeGraph_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_byEdges_withVertexIndexes_inducedSubgraph_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 }, { 0, 4, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 4, 3, 1.0 },
      } );
   // clang-format on

   const VectorType vertexIndexes( { 0, 1, 2, 3 } );
   const VectorType expectedDistances( { 0, 1, -1, -1, -1 } );
   VectorType distances;

   const auto allowUnitWeightOnly =
      [ = ] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType weight )
   {
      return weight == static_cast< typename GraphType::ValueType >( 1 ) && ! ( source == 1 && target == 2 );
   };

   TNL::Graphs::Algorithms::breadthFirstSearch(
      TNL::Graphs::makeSubGraph( graph, vertexIndexes, allowUnitWeightOnly ), 0, distances );

   ASSERT_EQ( distances, expectedDistances );
}

TYPED_TEST( GraphTest, test_BFS_byEdges_withVertexIndexes_inducedSubgraph )
{
   test_BFS_byEdges_withVertexIndexes_inducedSubgraph_impl< typename TestFixture::GraphType >();
}

TYPED_TEST( GraphTest, test_BFS_withInactiveStart_throws )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   const GraphType graph( 4, { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 } } );
   const VectorType vertexIndexes( { 0, 1, 2 } );
   VectorType distances;

   EXPECT_THROW(
      TNL::Graphs::Algorithms::breadthFirstSearch( TNL::Graphs::makeSubGraph( graph, vertexIndexes ), 3, distances ),
      std::invalid_argument );
}

// clang-format off
// Directed graph A (10 vertices, unit weights, symmetric adjacency).
// Used as the common "large" graph for subgraph cross-validation tests.
//
//     0---1---2
//     |   |   |
//     3---4---5
//     |   |   |
//     6---7---8---9
//
// Edges (both directions):
//   0-1, 0-3, 1-2, 1-4, 2-5, 3-4, 3-6, 4-5, 4-7, 5-8, 6-7, 7-8, 8-9
// clang-format on

template< typename GraphType >
GraphType
makeDirectedGraphA()
{
   // clang-format off
   return GraphType(
      10,
      {
         { 0, 1, 1 }, { 0, 3, 1 },
         { 1, 0, 1 }, { 1, 2, 1 }, { 1, 4, 1 },
         { 2, 1, 1 }, { 2, 5, 1 },
         { 3, 0, 1 }, { 3, 4, 1 }, { 3, 6, 1 },
         { 4, 1, 1 }, { 4, 3, 1 }, { 4, 5, 1 }, { 4, 7, 1 },
         { 5, 2, 1 }, { 5, 4, 1 }, { 5, 8, 1 },
         { 6, 3, 1 }, { 6, 7, 1 },
         { 7, 4, 1 }, { 7, 6, 1 }, { 7, 8, 1 },
         { 8, 5, 1 }, { 8, 7, 1 }, { 8, 9, 1 },
         { 9, 8, 1 },
      } );
   // clang-format on
}

// Subgraph B: graph A with vertices {2,5,8} removed.
// Remaining: {0,1,3,4,6,7,9} -> remapped to {0,1,2,3,4,5,6}
// oldToNew: 0->0, 1->1, 3->2, 4->3, 6->4, 7->5, 9->6
// newToOld: 0<-0, 1<-1, 2<-3, 3<-4, 4<-6, 5<-7, 6<-9
template< typename GraphType >
GraphType
makeSubgraphB_directed()
{
   // clang-format off
   return GraphType(
      7,
      {
         { 0, 1, 1 }, { 0, 2, 1 },
         { 1, 0, 1 }, { 1, 3, 1 },
         { 2, 0, 1 }, { 2, 3, 1 }, { 2, 4, 1 },
         { 3, 1, 1 }, { 3, 2, 1 }, { 3, 5, 1 },
         { 4, 2, 1 }, { 4, 5, 1 },
         { 5, 3, 1 }, { 5, 4, 1 },
      } );
   // clang-format on
}

// Subgraph D: graph A with cut-vertex {4} removed.
// Remaining: {0,1,2,3,5,6,7,8,9} -> remapped to {0,1,2,3,4,5,6,7,8}
// oldToNew: 0->0, 1->1, 2->2, 3->3, 5->4, 6->5, 7->6, 8->7, 9->8
template< typename GraphType >
GraphType
makeSubgraphD_directed()
{
   // clang-format off
   return GraphType(
      9,
      {
         { 0, 1, 1 }, { 0, 3, 1 },
         { 1, 0, 1 }, { 1, 2, 1 },
         { 2, 1, 1 }, { 2, 4, 1 },
         { 3, 0, 1 }, { 3, 5, 1 },
         { 4, 2, 1 }, { 4, 7, 1 },
         { 5, 3, 1 }, { 5, 6, 1 },
         { 6, 5, 1 }, { 6, 7, 1 },
         { 7, 4, 1 }, { 7, 6, 1 }, { 7, 8, 1 },
         { 8, 7, 1 },
      } );
   // clang-format on
}

// Subgraph C: graph A with edges {0,3} and {3,0} removed.
// All 10 vertices, just missing that one bidirectional edge.
template< typename GraphType >
GraphType
makeSubgraphC_directed()
{
   // clang-format off
   return GraphType(
      10,
      {
         { 0, 1, 1 },
         { 1, 0, 1 }, { 1, 2, 1 }, { 1, 4, 1 },
         { 2, 1, 1 }, { 2, 5, 1 },
         { 3, 4, 1 }, { 3, 6, 1 },
         { 4, 1, 1 }, { 4, 3, 1 }, { 4, 5, 1 }, { 4, 7, 1 },
         { 5, 2, 1 }, { 5, 4, 1 }, { 5, 8, 1 },
         { 6, 3, 1 }, { 6, 7, 1 },
         { 7, 4, 1 }, { 7, 6, 1 }, { 7, 8, 1 },
         { 8, 5, 1 }, { 8, 7, 1 }, { 8, 9, 1 },
         { 9, 8, 1 },
      } );
   // clang-format on
}

// Subgraph for E2: graph A restricted to vertices {0,1,3,4,6,7}
// with edges {0,3} and {3,0} also removed.
// Remap: 0->0, 1->1, 3->2, 4->3, 6->4, 7->5
template< typename GraphType >
GraphType
makeSubgraphE2_directed()
{
   // clang-format off
   return GraphType(
      6,
      {
         { 0, 1, 1 },
         { 1, 0, 1 }, { 1, 3, 1 },
         { 2, 3, 1 }, { 2, 4, 1 },
         { 3, 1, 1 }, { 3, 2, 1 }, { 3, 5, 1 },
         { 4, 2, 1 }, { 4, 5, 1 },
         { 5, 3, 1 }, { 5, 4, 1 },
      } );
   // clang-format on
}

template< typename VectorType >
void
remapAndCompareDistances( const VectorType& distA, const VectorType& distB, const std::vector< int >& newToOld )
{
   using IndexType = typename VectorType::IndexType;
   for( int i = 0; i < (int) newToOld.size(); i++ ) {
      IndexType expected = distB.getElement( i );
      IndexType actual = distA.getElement( newToOld[ i ] );
      ASSERT_EQ( actual, expected ) << "vertex " << newToOld[ i ] << " (subgraph idx " << i << ")";
   }
}

template< typename GraphType >
void
test_BFS_subgraph_vertex_removal_predicate_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphB = makeSubgraphB_directed< GraphType >();

   const auto excludeVertices = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 2 && v != 5 && v != 8;
   };

   VectorType distA, distB;
   TNL::Graphs::Algorithms::breadthFirstSearch( TNL::Graphs::makeSubGraph( graphA, excludeVertices ), 0, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphB, 0, distB );

   // oldToNew: 0->0, 1->1, 3->2, 4->3, 6->4, 7->5, 9->6
   // newToOld: 0, 1, 3, 4, 6, 7, 9
   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   remapAndCompareDistances( distA, distB, newToOld );

   ASSERT_EQ( distA.getElement( 2 ), -1 );
   ASSERT_EQ( distA.getElement( 5 ), -1 );
   ASSERT_EQ( distA.getElement( 8 ), -1 );
}

TYPED_TEST( GraphTest, test_BFS_subgraph_vertex_removal_predicate )
{
   test_BFS_subgraph_vertex_removal_predicate_impl< typename TestFixture::GraphType >();
}

TYPED_TEST( GraphTest, test_BFS_subgraph_vertex_removal_withVertexIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphB = makeSubgraphB_directed< GraphType >();

   const VectorType vertexIndexes( { 0, 1, 3, 4, 6, 7, 9 } );

   VectorType distA, distB;
   TNL::Graphs::Algorithms::breadthFirstSearch( TNL::Graphs::makeSubGraph( graphA, vertexIndexes ), 0, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphB, 0, distB );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   remapAndCompareDistances( distA, distB, newToOld );

   ASSERT_EQ( distA.getElement( 2 ), -1 );
   ASSERT_EQ( distA.getElement( 5 ), -1 );
   ASSERT_EQ( distA.getElement( 8 ), -1 );
}

template< typename GraphType >
void
test_BFS_subgraph_vertex_removal_disconnected_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphD = makeSubgraphD_directed< GraphType >();

   const auto excludeFour = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 4;
   };

   VectorType distA, distD;
   TNL::Graphs::Algorithms::breadthFirstSearch( TNL::Graphs::makeSubGraph( graphA, excludeFour ), 0, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphD, 0, distD );

   // oldToNew: 0->0, 1->1, 2->2, 3->3, 5->4, 6->5, 7->6, 8->7, 9->8
   // newToOld: 0, 1, 2, 3, 5, 6, 7, 8, 9
   const std::vector< int > newToOld = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
   remapAndCompareDistances( distA, distD, newToOld );

   ASSERT_EQ( distA.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_BFS_subgraph_vertex_removal_disconnected )
{
   test_BFS_subgraph_vertex_removal_disconnected_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_subgraph_edge_removal_wholeGraph_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphC = makeSubgraphC_directed< GraphType >();

   const auto blockEdge03 = [ = ] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType )
   {
      return ! ( source == 0 && target == 3 ) && ! ( source == 3 && target == 0 );
   };

   VectorType distA, distC;
   TNL::Graphs::Algorithms::breadthFirstSearch(
      TNL::Graphs::makeSubGraph( graphA, TNL::Graphs::edgeOnly, blockEdge03 ), 0, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphC, 0, distC );

   ASSERT_EQ( distA, distC );
}

TYPED_TEST( GraphTest, test_BFS_subgraph_edge_removal_wholeGraph )
{
   test_BFS_subgraph_edge_removal_wholeGraph_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_subgraph_edge_removal_withVertexIndexes_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphE2 = makeSubgraphE2_directed< GraphType >();

   const VectorType vertexIndexes( { 0, 1, 3, 4, 6, 7 } );
   const auto blockEdge03 = [ = ] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType )
   {
      return ! ( source == 0 && target == 3 ) && ! ( source == 3 && target == 0 );
   };

   VectorType distA, distE2;
   TNL::Graphs::Algorithms::breadthFirstSearch( TNL::Graphs::makeSubGraph( graphA, vertexIndexes, blockEdge03 ), 0, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphE2, 0, distE2 );

   // newToOld: 0, 1, 3, 4, 6, 7
   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7 };
   remapAndCompareDistances( distA, distE2, newToOld );

   ASSERT_EQ( distA.getElement( 2 ), -1 );
   ASSERT_EQ( distA.getElement( 5 ), -1 );
   ASSERT_EQ( distA.getElement( 8 ), -1 );
   ASSERT_EQ( distA.getElement( 9 ), -1 );
}

TYPED_TEST( GraphTest, test_BFS_subgraph_edge_removal_withVertexIndexes )
{
   test_BFS_subgraph_edge_removal_withVertexIndexes_impl< typename TestFixture::GraphType >();
}

// NEW#1: whole-graph BFS with an edge predicate and a visitor callback.
// Mirrors test_BFS_byEdges_wholeGraph (same graph + forbidOneToTwo) plus a
// visitor that records each visited node's distance into a view-backed vector.
template< typename GraphType >
void
test_BFS_withVisitor_edgePredicate_wholeGraph_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 }, { 0, 4, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 4, 3, 1.0 },
      } );
   // clang-format on

   // Edge 1->2 is blocked, so vertex 2 becomes unreachable and vertex 3 is
   // reached via 0->4->3 instead.
   const VectorType expectedDistances( { 0, 1, -1, 2, 1 } );
   // The visitor is not invoked on the start node, so entry 0 stays at -1.
   const VectorType expectedVisited( { -1, 1, -1, 2, 1 } );
   VectorType distances;
   VectorType visitedDistances( graph.getVertexCount(), -1 );
   auto visitedDistancesView = visitedDistances.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      visitedDistancesView[ vertex ] = distance;
   };
   const auto forbidOneToTwo =
      [ = ] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType weight )
   {
      return ! ( source == 1 && target == 2 );
   };

   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      TNL::Graphs::makeSubGraph( graph, TNL::Graphs::edgeOnly, forbidOneToTwo ), 0, visitor, distances );

   ASSERT_EQ( distances, expectedDistances );
   EXPECT_EQ( visitedDistances, expectedVisited );
}

TYPED_TEST( GraphTest, test_BFS_withVisitor_edgePredicate_wholeGraph )
{
   test_BFS_withVisitor_edgePredicate_wholeGraph_impl< typename TestFixture::GraphType >();
}

// NEW#2: vertex-indexed subgraph BFS with an edge predicate and a visitor callback.
// Mirrors test_BFS_subgraph_edge_removal_withVertexIndexes (same graph A, subgraph E2,
// vertex indexes, and blockEdge03) plus a visitor, cross-validating distances
// against the materialized subgraph.
template< typename GraphType >
void
test_BFS_withVisitor_edgePredicate_subgraph_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphE2 = makeSubgraphE2_directed< GraphType >();

   const VectorType vertexIndexes( { 0, 1, 3, 4, 6, 7 } );
   const auto blockEdge03 = [ = ] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType )
   {
      return ! ( source == 0 && target == 3 ) && ! ( source == 3 && target == 0 );
   };

   VectorType distA, distE2;
   VectorType visitedDistances( graphA.getVertexCount(), -1 );
   auto visitedDistancesView = visitedDistances.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      visitedDistancesView[ vertex ] = distance;
   };

   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      TNL::Graphs::makeSubGraph( graphA, vertexIndexes, blockEdge03 ), 0, visitor, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphE2, 0, distE2 );

   // newToOld: 0, 1, 3, 4, 6, 7
   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7 };
   remapAndCompareDistances( distA, distE2, newToOld );

   ASSERT_EQ( distA.getElement( 2 ), -1 );
   ASSERT_EQ( distA.getElement( 5 ), -1 );
   ASSERT_EQ( distA.getElement( 8 ), -1 );
   ASSERT_EQ( distA.getElement( 9 ), -1 );

   // Visitor records the distance of every visited node; the start node (0) is
   // not passed to the visitor and excluded vertices stay at -1.
   const VectorType expectedVisited( { -1, 1, -1, 3, 2, -1, 4, 3, -1, -1 } );
   EXPECT_EQ( visitedDistances, expectedVisited );
}

TYPED_TEST( GraphTest, test_BFS_withVisitor_edgePredicate_subgraph )
{
   test_BFS_withVisitor_edgePredicate_subgraph_impl< typename TestFixture::GraphType >();
}

// NEW#3: predicate-induced subgraph BFS with an edge predicate and a visitor.
// Excluding vertices {2,5,8} and blocking edge 0-3 yields the same reachable
// set as subgraph E2 (vertices {0,1,3,4,6,7} with edge 0-3 removed), so the
// distances are cross-validated against it. Vertex 9 is active but unreachable
// (its only edge goes to the excluded vertex 8).
template< typename GraphType >
void
test_BFS_ifWithVisitor_edgePredicate_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphE2 = makeSubgraphE2_directed< GraphType >();

   const auto excludeVertices = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 2 && v != 5 && v != 8;
   };
   const auto blockEdge03 = [ = ] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType )
   {
      return ! ( source == 0 && target == 3 ) && ! ( source == 3 && target == 0 );
   };

   VectorType distA, distE2;
   VectorType visitedDistances( graphA.getVertexCount(), -1 );
   auto visitedDistancesView = visitedDistances.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      visitedDistancesView[ vertex ] = distance;
   };

   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      TNL::Graphs::makeSubGraph( graphA, excludeVertices, blockEdge03 ), 0, visitor, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphE2, 0, distE2 );

   // newToOld: 0, 1, 3, 4, 6, 7
   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7 };
   remapAndCompareDistances( distA, distE2, newToOld );

   ASSERT_EQ( distA.getElement( 2 ), -1 );
   ASSERT_EQ( distA.getElement( 5 ), -1 );
   ASSERT_EQ( distA.getElement( 8 ), -1 );
   ASSERT_EQ( distA.getElement( 9 ), -1 );  // active but unreachable (only edge goes to excluded 8)

   const VectorType expectedVisited( { -1, 1, -1, 3, 2, -1, 4, 3, -1, -1 } );
   EXPECT_EQ( visitedDistances, expectedVisited );
}

TYPED_TEST( GraphTest, test_BFS_ifWithVisitor_edgePredicate )
{
   test_BFS_ifWithVisitor_edgePredicate_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_predecessors_basic_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   //   0 → 1 → 2
   //   ↓   ↓
   //   3   4
   //   ↓
   //   5
   GraphType graph(
      6,
      {
         { 0, 1, 1.0 }, { 0, 3, 1.0 },
         { 1, 2, 1.0 }, { 1, 4, 1.0 },
         { 3, 5, 1.0 },
      } );
   // clang-format on

   for( IndexType start = 0; start < graph.getVertexCount(); ++start ) {
      VectorType distances;
      VectorType predecessors;
      TNL::Graphs::Algorithms::breadthFirstSearchWithPredecessors( graph, start, distances, predecessors );

      // Check predecessor invariants: distances[pred[v]] == distances[v] - 1,
      // and pred[v] == -1 for start and unreachable vertices.
      for( IndexType v = 0; v < graph.getVertexCount(); ++v ) {
         if( v == start || distances.getElement( v ) == -1 ) {
            EXPECT_EQ( predecessors.getElement( v ), -1 ) << "start=" << start << " v=" << v;
         }
         else {
            IndexType pred = predecessors.getElement( v );
            EXPECT_GE( pred, 0 ) << "start=" << start << " v=" << v;
            EXPECT_EQ( distances.getElement( pred ), distances.getElement( v ) - 1 ) << "start=" << start << " v=" << v
               << " pred=" << pred;
         }
      }
   }
}

TYPED_TEST( GraphTest, test_BFS_predecessors_basic )
{
   test_BFS_predecessors_basic_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_predecessors_deterministic_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   //   0 → 1
   //   0 → 2     (both 1 and 2 are at distance 1, so deterministic mode
   //   1 → 3      should pick the smallest source for each target)
   //   2 → 3
   GraphType graph(
      4,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   // Expected deterministic predecessors from source 0:
   //   pred[0] = -1 (start)
   //   pred[1] = 0  (only edge to 1 is from 0)
   //   pred[2] = 0  (only edge to 2 is from 0)
   //   pred[3] = 1  (smallest source among {1, 2} that reaches 3)
   const VectorType expectedPredecessors( { -1, 0, 0, 1 } );
   const VectorType expectedDistances( { 0, 1, 1, 2 } );

   // Run multiple times — deterministic mode must produce identical results
   for( int run = 0; run < 5; ++run ) {
      VectorType distances;
      VectorType predecessors;
      TNL::Graphs::Algorithms::breadthFirstSearchWithPredecessors( graph, 0, distances, predecessors, true );

      ASSERT_EQ( distances, expectedDistances ) << "run=" << run;
      ASSERT_EQ( predecessors, expectedPredecessors ) << "run=" << run;
   }
}

TYPED_TEST( GraphTest, test_BFS_predecessors_deterministic )
{
   test_BFS_predecessors_deterministic_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_predecessors_unreachable_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   //   0 → 1     2 → 3    (two disconnected components)
   GraphType graph(
      4,
      {
         { 0, 1, 1.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   VectorType distances;
   VectorType predecessors;
   TNL::Graphs::Algorithms::breadthFirstSearchWithPredecessors( graph, 0, distances, predecessors );

   const VectorType expectedDistances( { 0, 1, -1, -1 } );
   ASSERT_EQ( distances, expectedDistances );

   // Unreachable vertices must have predecessor -1
   EXPECT_EQ( predecessors.getElement( 0 ), -1 );  // start
   EXPECT_EQ( predecessors.getElement( 1 ), 0 );  // parent of 1 is 0
   EXPECT_EQ( predecessors.getElement( 2 ), -1 );  // unreachable
   EXPECT_EQ( predecessors.getElement( 3 ), -1 );  // unreachable
}

TYPED_TEST( GraphTest, test_BFS_predecessors_unreachable )
{
   test_BFS_predecessors_unreachable_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_visitor_called_once_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   //   0 → 1
   //   0 → 2     (vertex 3 is reachable from both 1 and 2 — visitor must
   //   1 → 3      be called exactly once for it)
   //   2 → 3
   GraphType graph(
      4,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   // Count visitor calls per vertex using a distances vector
   VectorType visitCount( graph.getVertexCount(), 0 );
   auto visitCountView = visitCount.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      (void) distance;
      visitCountView[ vertex ] += 1;
   };

   VectorType distances;
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor( graph, 0, visitor, distances );

   // Visitor is not called for the start vertex (distance 0)
   EXPECT_EQ( visitCount.getElement( 0 ), 0 );
   // Each other reachable vertex must be visited exactly once
   EXPECT_EQ( visitCount.getElement( 1 ), 1 );
   EXPECT_EQ( visitCount.getElement( 2 ), 1 );
   EXPECT_EQ( visitCount.getElement( 3 ), 1 );
}

TYPED_TEST( GraphTest, test_BFS_visitor_called_once )
{
   test_BFS_visitor_called_once_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_BFS_visitor_with_predecessors_impl()
{
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   //   0 → 1 → 3
   //   0 → 2 → 3
   GraphType graph(
      4,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   VectorType visitedDistances( graph.getVertexCount(), -1 );
   auto visitedDistancesView = visitedDistances.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      visitedDistancesView[ vertex ] = distance;
   };

   VectorType distances;
   VectorType predecessors;
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitorAndPredecessors( graph, 0, visitor, distances, predecessors, true );

   const VectorType expectedDistances( { 0, 1, 1, 2 } );
   ASSERT_EQ( distances, expectedDistances );

   // Visitor should record distances for all reachable vertices except start
   const VectorType expectedVisited( { -1, 1, 1, 2 } );
   EXPECT_EQ( visitedDistances, expectedVisited );

   // Predecessor invariants
   for( IndexType v = 0; v < graph.getVertexCount(); ++v ) {
      if( v == 0 || distances.getElement( v ) == -1 ) {
         EXPECT_EQ( predecessors.getElement( v ), -1 );
      }
      else {
         IndexType pred = predecessors.getElement( v );
         EXPECT_GE( pred, 0 );
         EXPECT_EQ( distances.getElement( pred ), distances.getElement( v ) - 1 );
      }
   }
}

TYPED_TEST( GraphTest, test_BFS_visitor_with_predecessors )
{
   test_BFS_visitor_with_predecessors_impl< typename TestFixture::GraphType >();
}

// Force top-down bitmap mode (threshold = 1.0 → every iteration uses forAllEdges) and
// verify that distances match the compact-mode reference for all start vertices.
template< typename GraphType >
void
test_BFS_bitmap_distances_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graph = makeDirectedGraphA< GraphType >();

   for( IndexType start = 0; start < graph.getVertexCount(); ++start ) {
      VectorType distCompact, distBitmap;
      TNL::Graphs::Algorithms::breadthFirstSearch( graph, start, distCompact );
      TNL::Graphs::Algorithms::breadthFirstSearch( graph, start, distBitmap, 1.0 );
      ASSERT_EQ( distBitmap, distCompact ) << "start=" << start;
   }
}

TYPED_TEST( GraphTest, test_BFS_bitmap_distances )
{
   test_BFS_bitmap_distances_impl< typename TestFixture::GraphType >();
}

// Force top-down bitmap mode + deterministic predecessors + visitor, verify correctness.
template< typename GraphType >
void
test_BFS_bitmap_predecessors_deterministic_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   // clang-format off
   //   0 → 1
   //   0 → 2     (both 1 and 2 at distance 1; vertex 3 reachable from both)
   //   1 → 3
   //   2 → 3
   GraphType graph(
      4,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   const VectorType expectedPredecessors( { -1, 0, 0, 1 } );
   const VectorType expectedDistances( { 0, 1, 1, 2 } );

   VectorType visitedDistances( graph.getVertexCount(), -1 );
   auto visitedDistancesView = visitedDistances.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      visitedDistancesView[ vertex ] = distance;
   };

   VectorType distances, predecessors;
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitorAndPredecessors(
      graph, 0, visitor, distances, predecessors, true, 1.0 );

   ASSERT_EQ( distances, expectedDistances );
   ASSERT_EQ( predecessors, expectedPredecessors );

   const VectorType expectedVisited( { -1, 1, 1, 2 } );
   EXPECT_EQ( visitedDistances, expectedVisited );
}

TYPED_TEST( GraphTest, test_BFS_bitmap_predecessors_deterministic )
{
   test_BFS_bitmap_predecessors_deterministic_impl< typename TestFixture::GraphType >();
}

// Force top-down bitmap mode and verify the visitor is called exactly once per vertex.
template< typename GraphType >
void
test_BFS_bitmap_visitor_called_once_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   // clang-format off
   GraphType graph(
      4,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   VectorType visitCount( graph.getVertexCount(), 0 );
   auto visitCountView = visitCount.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      (void) distance;
      visitCountView[ vertex ] += 1;
   };

   VectorType distances;
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor( graph, 0, visitor, distances, 1.0 );

   EXPECT_EQ( visitCount.getElement( 0 ), 0 );
   EXPECT_EQ( visitCount.getElement( 1 ), 1 );
   EXPECT_EQ( visitCount.getElement( 2 ), 1 );
   EXPECT_EQ( visitCount.getElement( 3 ), 1 );
}

TYPED_TEST( GraphTest, test_BFS_bitmap_visitor_called_once )
{
   test_BFS_bitmap_visitor_called_once_impl< typename TestFixture::GraphType >();
}

// ---------------------------------------------------------------------------
// Undirected graph tests — bottom-up BFS direction optimization
// ---------------------------------------------------------------------------

template< typename Matrix >
class GraphTestUndirected : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::UndirectedGraph >;
};

using GraphTestUndirectedTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Sequential, int >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int >
#elif defined( __CUDACC__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int >
#endif
   >;

TYPED_TEST_SUITE( GraphTestUndirected, GraphTestUndirectedTypes );

// Undirected version of the 5-node graph from test_BFS_small.
// Edges are given once; the adjacency matrix mirrors them automatically.
//   0 --- 1 --- 2
//   |     |     |
//   3 --- 4 --- 5
// Distances from vertex 0: {0, 1, 2, 1, 2, 3}
template< typename GraphType >
void
test_BFS_undirected_small_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   // clang-format off
   GraphType graph(
      6,
      {
         { 0, 1, 1.0 }, { 0, 3, 1.0 },
         { 1, 2, 1.0 }, { 1, 4, 1.0 },
         { 2, 5, 1.0 },
         { 3, 4, 1.0 },
         { 4, 5, 1.0 },
      } );
   // clang-format on

   VectorType distances;
   TNL::Graphs::Algorithms::breadthFirstSearch( graph, 0, distances );

   const VectorType expectedDistances( { 0, 1, 2, 1, 2, 3 } );
   ASSERT_EQ( distances, expectedDistances );
}

TYPED_TEST( GraphTestUndirected, test_BFS_undirected_small )
{
   test_BFS_undirected_small_impl< typename TestFixture::GraphType >();
}

// Force bottom-up mode (threshold = 1.0 → every iteration uses bottom-up) and
// verify that distances match the compact-mode reference for all start vertices.
template< typename GraphType >
void
test_BFS_bottomup_distances_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   // clang-format off
   //   0 --- 1 --- 2
   //   |     |     |
   //   3 --- 4 --- 5
   GraphType graph(
      6,
      {
         { 0, 1, 1.0 }, { 0, 3, 1.0 },
         { 1, 2, 1.0 }, { 1, 4, 1.0 },
         { 2, 5, 1.0 },
         { 3, 4, 1.0 },
         { 4, 5, 1.0 },
      } );
   // clang-format on

   for( IndexType start = 0; start < graph.getVertexCount(); ++start ) {
      VectorType distCompact, distBottomUp;
      TNL::Graphs::Algorithms::breadthFirstSearch( graph, start, distCompact );
      TNL::Graphs::Algorithms::breadthFirstSearch( graph, start, distBottomUp, 0.0, 1.0 );
      ASSERT_EQ( distBottomUp, distCompact ) << "start=" << start;
   }
}

TYPED_TEST( GraphTestUndirected, test_BFS_bottomup_distances )
{
   test_BFS_bottomup_distances_impl< typename TestFixture::GraphType >();
}

// Force bottom-up mode + deterministic predecessors + visitor.
template< typename GraphType >
void
test_BFS_bottomup_predecessors_deterministic_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   // clang-format off
   //   0 --- 1
   //   |     |
   //   2 --- 3
   GraphType graph(
      4,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   const VectorType expectedDistances( { 0, 1, 1, 2 } );
   // Deterministic: smallest source wins. From vertex 0:
   //   pred[1] = 0 (only edge to 1 is from 0)
   //   pred[2] = 0 (only edge to 2 is from 0)
   //   pred[3] = 1 (smallest among {1, 2})
   const VectorType expectedPredecessors( { -1, 0, 0, 1 } );

   VectorType visitedDistances( graph.getVertexCount(), -1 );
   auto visitedDistancesView = visitedDistances.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      visitedDistancesView[ vertex ] = distance;
   };

   VectorType distances, predecessors;
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitorAndPredecessors(
      graph, 0, visitor, distances, predecessors, true, 0.0, 1.0 );

   ASSERT_EQ( distances, expectedDistances );
   ASSERT_EQ( predecessors, expectedPredecessors );

   const VectorType expectedVisited( { -1, 1, 1, 2 } );
   EXPECT_EQ( visitedDistances, expectedVisited );
}

TYPED_TEST( GraphTestUndirected, test_BFS_bottomup_predecessors_deterministic )
{
   test_BFS_bottomup_predecessors_deterministic_impl< typename TestFixture::GraphType >();
}

// Force bottom-up mode and verify the visitor is called exactly once per vertex.
template< typename GraphType >
void
test_BFS_bottomup_visitor_called_once_impl()
{
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   // clang-format off
   GraphType graph(
      4,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   VectorType visitCount( graph.getVertexCount(), 0 );
   auto visitCountView = visitCount.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      (void) distance;
      visitCountView[ vertex ] += 1;
   };

   VectorType distances;
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor( graph, 0, visitor, distances, 0.0, 1.0 );

   EXPECT_EQ( visitCount.getElement( 0 ), 0 );
   EXPECT_EQ( visitCount.getElement( 1 ), 1 );
   EXPECT_EQ( visitCount.getElement( 2 ), 1 );
   EXPECT_EQ( visitCount.getElement( 3 ), 1 );
}

TYPED_TEST( GraphTestUndirected, test_BFS_bottomup_visitor_called_once )
{
   test_BFS_bottomup_visitor_called_once_impl< typename TestFixture::GraphType >();
}

#include "../../main.h"
