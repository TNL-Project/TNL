#include <TNL/Graphs/Algorithms/breadthFirstSearch.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Containers/StaticVector.h>

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

   for( int start_node = 0; start_node < graph.getVertexCount(); ++start_node ) {
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

   for( int start_node = 0; start_node < graph.getVertexCount(); ++start_node ) {
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

   for( int start_node = 0; start_node < graph.getVertexCount(); start_node++ ) {
      TNL::Graphs::Algorithms::breadthFirstSearch( graph, start_node, distances );
      ASSERT_EQ( distances, expectedDistances[ start_node ] ) << "start_node: " << start_node;
   }
}

TYPED_TEST( GraphTest, test_BFS_withIndexes_inducedSubgraph )
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

   TNL::Graphs::Algorithms::breadthFirstSearch( graph, 0, vertexIndexes, distances );

   ASSERT_EQ( distances, expectedDistances );
}

TYPED_TEST( GraphTest, test_BFSIf_inducedSubgraph )
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
   const VectorType expectedDistances( { 0, 1, 2, -1, -1 } );
   VectorType distances;
   const auto firstThreeVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex <= 2;
   };

   TNL::Graphs::Algorithms::breadthFirstSearchIf( graph, 0, firstThreeVertices, distances );

   ASSERT_EQ( distances, expectedDistances );
}

TYPED_TEST( GraphTest, test_BFS_withIndexes_visitor )
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
   const VectorType vertexIndexes( { 0, 1, 2 } );
   const VectorType expectedDistances( { 0, 1, 2, -1, -1 } );
   VectorType distances;
   VectorType visitedDistances( graph.getVertexCount(), -1 );
   auto visitedDistancesView = visitedDistances.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      visitedDistancesView[ vertex ] = distance;
   };

   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor( graph, 0, vertexIndexes, visitor, distances );

   ASSERT_EQ( distances, expectedDistances );
   EXPECT_EQ( visitedDistances.getElement( 0 ), -1 );
   EXPECT_EQ( visitedDistances.getElement( 1 ), 1 );
   EXPECT_EQ( visitedDistances.getElement( 2 ), 2 );
   EXPECT_EQ( visitedDistances.getElement( 3 ), -1 );
   EXPECT_EQ( visitedDistances.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_BFS_byEdges_wholeGraph )
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

   const VectorType expectedDistances( { 0, 1, -1, 2, 1 } );
   VectorType distances;
   const auto forbidOneToTwo =
      [ = ] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType weight )
   {
      return ! ( source == 1 && target == 2 );
   };

   TNL::Graphs::Algorithms::breadthFirstSearch( graph, 0, forbidOneToTwo, distances );

   ASSERT_EQ( distances, expectedDistances );
}

TYPED_TEST( GraphTest, test_BFS_byEdges_withIndexes_inducedSubgraph )
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

   const VectorType vertexIndexes( { 0, 1, 2, 3 } );
   const VectorType expectedDistances( { 0, 1, -1, -1, -1 } );
   VectorType distances;

   const auto allowUnitWeightOnly =
      [ = ] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType weight )
   {
      return weight == static_cast< typename GraphType::ValueType >( 1 ) && ! ( source == 1 && target == 2 );
   };

   TNL::Graphs::Algorithms::breadthFirstSearch( graph, 0, vertexIndexes, allowUnitWeightOnly, distances );

   ASSERT_EQ( distances, expectedDistances );
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

   EXPECT_THROW( TNL::Graphs::Algorithms::breadthFirstSearch( graph, 3, vertexIndexes, distances ), std::invalid_argument );
}

// clang-format off
// Directed graph A (10 vertices, unit weights, symmetric adjacency).
// Used as the common "large" graph for subgraph cross-validation tests.
//
//     0---1---2---5---8---9
//     |   |   |   |   |
//     3---4---+   6---7
//     |   |       |   |
//     +---6-------+---+
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
remapAndCompareDistances(
   const VectorType& distA,
   const VectorType& distB,
   const std::vector< int >& newToOld,
   int origSize )
{
   using IndexType = typename VectorType::IndexType;
   for( int i = 0; i < (int) newToOld.size(); i++ ) {
      IndexType expected = distB.getElement( i );
      IndexType actual = distA.getElement( newToOld[ i ] );
      ASSERT_EQ( actual, expected ) << "vertex " << newToOld[ i ] << " (subgraph idx " << i << ")";
   }
}

TYPED_TEST( GraphTest, test_BFS_subgraph_vertex_removal_predicate )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphB = makeSubgraphB_directed< GraphType >();

   const auto excludeVertices = [=] __cuda_callable__( IndexType v )
   {
      return v != 2 && v != 5 && v != 8;
   };

   VectorType distA, distB;
   TNL::Graphs::Algorithms::breadthFirstSearchIf( graphA, 0, excludeVertices, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphB, 0, distB );

   // oldToNew: 0->0, 1->1, 3->2, 4->3, 6->4, 7->5, 9->6
   // newToOld: 0, 1, 3, 4, 6, 7, 9
   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   remapAndCompareDistances( distA, distB, newToOld, 10 );

   ASSERT_EQ( distA.getElement( 2 ), -1 );
   ASSERT_EQ( distA.getElement( 5 ), -1 );
   ASSERT_EQ( distA.getElement( 8 ), -1 );
}

TYPED_TEST( GraphTest, test_BFS_subgraph_vertex_removal_indexed )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphB = makeSubgraphB_directed< GraphType >();

   const VectorType vertexIndexes( { 0, 1, 3, 4, 6, 7, 9 } );

   VectorType distA, distB;
   TNL::Graphs::Algorithms::breadthFirstSearch( graphA, 0, vertexIndexes, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphB, 0, distB );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   remapAndCompareDistances( distA, distB, newToOld, 10 );

   ASSERT_EQ( distA.getElement( 2 ), -1 );
   ASSERT_EQ( distA.getElement( 5 ), -1 );
   ASSERT_EQ( distA.getElement( 8 ), -1 );
}

TYPED_TEST( GraphTest, test_BFS_subgraph_vertex_removal_disconnected )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphD = makeSubgraphD_directed< GraphType >();

   const auto excludeFour = [=] __cuda_callable__( IndexType v )
   {
      return v != 4;
   };

   VectorType distA, distD;
   TNL::Graphs::Algorithms::breadthFirstSearchIf( graphA, 0, excludeFour, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphD, 0, distD );

   // oldToNew: 0->0, 1->1, 2->2, 3->3, 5->4, 6->5, 7->6, 8->7, 9->8
   // newToOld: 0, 1, 2, 3, 5, 6, 7, 8, 9
   const std::vector< int > newToOld = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
   remapAndCompareDistances( distA, distD, newToOld, 10 );

   ASSERT_EQ( distA.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_BFS_subgraph_edge_removal_wholeGraph )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphC = makeSubgraphC_directed< GraphType >();

   const auto blockEdge03 = [=] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType )
   {
      return ! ( source == 0 && target == 3 ) && ! ( source == 3 && target == 0 );
   };

   VectorType distA, distC;
   TNL::Graphs::Algorithms::breadthFirstSearch( graphA, 0, blockEdge03, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphC, 0, distC );

   ASSERT_EQ( distA, distC );
}

TYPED_TEST( GraphTest, test_BFS_subgraph_edge_removal_withIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeDirectedGraphA< GraphType >();
   const auto subgraphE2 = makeSubgraphE2_directed< GraphType >();

   const VectorType vertexIndexes( { 0, 1, 3, 4, 6, 7 } );
   const auto blockEdge03 = [=] __cuda_callable__( IndexType source, IndexType target, typename GraphType::ValueType )
   {
      return ! ( source == 0 && target == 3 ) && ! ( source == 3 && target == 0 );
   };

   VectorType distA, distE2;
   TNL::Graphs::Algorithms::breadthFirstSearch( graphA, 0, vertexIndexes, blockEdge03, distA );
   TNL::Graphs::Algorithms::breadthFirstSearch( subgraphE2, 0, distE2 );

   // newToOld: 0, 1, 3, 4, 6, 7
   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7 };
   remapAndCompareDistances( distA, distE2, newToOld, 10 );

   ASSERT_EQ( distA.getElement( 2 ), -1 );
   ASSERT_EQ( distA.getElement( 5 ), -1 );
   ASSERT_EQ( distA.getElement( 8 ), -1 );
   ASSERT_EQ( distA.getElement( 9 ), -1 );
}

#include "../../main.h"
