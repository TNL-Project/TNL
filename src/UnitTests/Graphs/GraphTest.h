// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Graphs/Graph.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Vector.h>

#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class GraphBasicTest : public ::testing::Test
{
protected:
   using AdjacencyMatrixType = Matrix;
   using DirectedGraphType = TNL::Graphs::Graph< AdjacencyMatrixType, TNL::Graphs::DirectedGraph >;
   using UndirectedGraphType = TNL::Graphs::Graph< AdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
};

// types for which GraphBasicTest is instantiated
using GraphBasicTestTypes = ::testing::Types< TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int >,
                                              TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >,
                                              TNL::Matrices::DenseMatrix< double, TNL::Devices::Sequential, int >,
                                              TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int >
#if defined( __CUDACC__ )
                                              ,
                                              TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >,
                                              TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
                                              ,
                                              TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int >,
                                              TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int >
#endif
                                              >;

TYPED_TEST_SUITE( GraphBasicTest, GraphBasicTestTypes );

TYPED_TEST( GraphBasicTest, DefaultConstructor )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph;
   EXPECT_EQ( graph.getNodeCount(), 0 );
   EXPECT_EQ( graph.getEdgeCount(), 0 );
}

TYPED_TEST( GraphBasicTest, GraphTypeChecks )
{
   using DirectedGraphType = typename TestFixture::DirectedGraphType;
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;

   EXPECT_TRUE( DirectedGraphType::isDirected() );
   EXPECT_FALSE( DirectedGraphType::isUndirected() );

   EXPECT_FALSE( UndirectedGraphType::isDirected() );
   EXPECT_TRUE( UndirectedGraphType::isUndirected() );
}

TYPED_TEST( GraphBasicTest, SetNodeCount )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   GraphType graph;
   graph.setNodeCount( 5 );
   EXPECT_EQ( graph.getNodeCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 0 );

   graph.setNodeCount( 10 );
   EXPECT_EQ( graph.getNodeCount(), 10 );

   auto graph_view = graph.getView();
   EXPECT_EQ( graph_view.getNodeCount(), 10 );

   using GraphViewType = typename GraphType::ViewType;
   EXPECT_TRUE( ( std::is_same_v< GraphViewType, decltype( graph.getView() ) > ) );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;

      SymmetricGraphType symmetricGraph;
      symmetricGraph.setNodeCount( 5 );
      EXPECT_EQ( symmetricGraph.getNodeCount(), 5 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 0 );

      auto symmetric_graph_view = symmetricGraph.getView();
      EXPECT_EQ( symmetric_graph_view.getNodeCount(), 5 );
      EXPECT_EQ( symmetric_graph_view.getEdgeCount(), 0 );

      EXPECT_TRUE(
         ( std::is_same_v< typename SymmetricGraphType::ConstViewType, decltype( symmetricGraph.getConstView() ) > ) );

      auto const_symmetric_graph_view = symmetricGraph.getConstView();
      EXPECT_EQ( const_symmetric_graph_view.getNodeCount(), 5 );
      EXPECT_EQ( const_symmetric_graph_view.getEdgeCount(), 0 );
   }
}

TYPED_TEST( GraphBasicTest, ConstructorWithInitializerList )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   // Create a simple directed graph with 4 nodes and 5 edges
   GraphType graph( 4, { { 0, 1, 1.0 }, { 0, 2, 2.0 }, { 1, 2, 3.0 }, { 1, 3, 4.0 }, { 2, 3, 5.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 4 );
   EXPECT_EQ( graph.getEdgeCount(), 5 );

   // Check if the edges are stored correctly in the adjacency matrix
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 3.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 3 ), 4.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 3 ), 5.0 );

   // Verify that non-existing edges have weight 0
   EXPECT_EQ( graph.getEdgeWeight( 1, 0 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 0 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 3, 0 ), 0.0 );

   // Create the same graph as undirected
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;
   UndirectedGraphType undirectedGraph( 4, { { 0, 1, 1.0 }, { 0, 2, 2.0 }, { 1, 2, 3.0 }, { 1, 3, 4.0 }, { 2, 3, 5.0 } } );

   EXPECT_EQ( undirectedGraph.getNodeCount(), 4 );
   EXPECT_EQ( undirectedGraph.getEdgeCount(), 5 );

   // Check if the edges are symmetric in the adjacency matrix for undirected graph
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 0 ), 1.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 0, 2 ), 2.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 0 ), 2.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 2 ), 3.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 1 ), 3.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 3 ), 4.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 3, 1 ), 4.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 3 ), 5.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 3, 2 ), 5.0 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      SymmetricGraphType symmetricGraph( 4, { { 0, 1, 1.0 }, { 0, 2, 2.0 }, { 1, 2, 3.0 }, { 1, 3, 4.0 }, { 2, 3, 5.0 } } );
      EXPECT_EQ( symmetricGraph.getNodeCount(), 4 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 5 );

      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 1 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 0 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 2 ), 2.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 0 ), 2.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 2 ), 3.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 1 ), 3.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 3 ), 4.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 3, 1 ), 4.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 3 ), 5.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 3, 2 ), 5.0 );
   }
}

TYPED_TEST( GraphBasicTest, ConstructorWithMap )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   std::map< std::pair< IndexType, IndexType >, ValueType > edges;
   edges[ { 0, 1 } ] = 1.0;
   edges[ { 0, 2 } ] = 2.0;
   edges[ { 1, 2 } ] = 3.0;
   edges[ { 1, 3 } ] = 4.0;
   edges[ { 2, 3 } ] = 5.0;

   GraphType graph( 4, edges );

   EXPECT_EQ( graph.getNodeCount(), 4 );
   EXPECT_EQ( graph.getEdgeCount(), 5 );

   // Check if the edges are stored correctly in the adjacency matrix
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 3.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 3 ), 4.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 3 ), 5.0 );

   // Create the same graph as undirected
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;
   UndirectedGraphType undirectedGraph( 4, edges );

   EXPECT_EQ( undirectedGraph.getNodeCount(), 4 );
   EXPECT_EQ( undirectedGraph.getEdgeCount(), 5 );

   // Check if the edges are symmetric in the adjacency matrix for undirected graph
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 0 ), 1.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 0, 2 ), 2.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 0 ), 2.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 2 ), 3.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 1 ), 3.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 3 ), 4.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 3, 1 ), 4.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 3 ), 5.0 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 3, 2 ), 5.0 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      SymmetricGraphType symmetricGraph( 4, edges );
      EXPECT_EQ( symmetricGraph.getNodeCount(), 4 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 5 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 1 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 0 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 2 ), 2.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 0 ), 2.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 2 ), 3.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 1 ), 3.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 3 ), 4.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 3, 1 ), 4.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 3 ), 5.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 3, 2 ), 5.0 );
   }
}

TYPED_TEST( GraphBasicTest, SetEdges )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   GraphType graph;
   graph.setNodeCount( 3 );

   std::map< std::pair< IndexType, IndexType >, ValueType > edges;
   edges[ { 0, 1 } ] = 1.5;
   edges[ { 1, 2 } ] = 2.5;
   edges[ { 0, 2 } ] = 3.5;

   graph.setEdges( edges );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 3 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.5 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 2.5 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 3.5 );

   // Test for undirected graph
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;
   UndirectedGraphType undirectedGraph;
   undirectedGraph.setNodeCount( 3 );
   undirectedGraph.setEdges( edges );

   EXPECT_EQ( undirectedGraph.getNodeCount(), 3 );
   EXPECT_EQ( undirectedGraph.getEdgeCount(), 3 );

   EXPECT_EQ( undirectedGraph.getEdgeWeight( 0, 1 ), 1.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 0 ), 1.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 2 ), 2.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 1 ), 2.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 0, 2 ), 3.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 0 ), 3.5 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      SymmetricGraphType symmetricGraph( 3 );
      symmetricGraph.setEdges( edges );
      EXPECT_EQ( symmetricGraph.getNodeCount(), 3 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 3 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 1 ), 1.5 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 0 ), 1.5 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 2 ), 2.5 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 1 ), 2.5 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 2 ), 3.5 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 0 ), 3.5 );
   }
}

TYPED_TEST( GraphBasicTest, GetAdjacencyMatrix )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );

   const auto& matrix = graph.getAdjacencyMatrix();
   EXPECT_EQ( matrix.getRows(), 3 );
   EXPECT_EQ( matrix.getColumns(), 3 );
   EXPECT_EQ( matrix.getNonzeroElementsCount(), 2 );
}

TYPED_TEST( GraphBasicTest, SetAdjacencyMatrix )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;

   AdjacencyMatrixType matrix;
   matrix.setDimensions( 3, 3 );

   std::map< std::pair< IndexType, IndexType >, ValueType > elements;
   elements[ { 0, 1 } ] = 1.0;
   elements[ { 1, 2 } ] = 2.0;
   elements[ { 2, 0 } ] = 3.0;
   matrix.setElements( elements );

   GraphType graph;
   graph.setAdjacencyMatrix( matrix );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 3 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 0 ), 3.0 );
}

TYPED_TEST( GraphBasicTest, CopyConstructor )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 } } );

   GraphType graph2( graph1 );

   EXPECT_EQ( graph2.getNodeCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 3 );
   EXPECT_EQ( graph1, graph2 );

   EXPECT_EQ( graph1.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph1.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph1.getEdgeWeight( 0, 2 ), 3.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 0, 2 ), 3.0 );

   // Test assignment between directed and undirected graphs
   using DirectedGraphType = typename TestFixture::DirectedGraphType;
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;

   UndirectedGraphType undirectedGraph( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 } } );
   DirectedGraphType directedGraph;

   directedGraph = undirectedGraph;

   EXPECT_EQ( directedGraph.getNodeCount(), 3 );
   EXPECT_EQ( directedGraph.getEdgeCount(), 6 );  // because undirected edges become directed edges in both directions

   // Check that directed graph has symmetric edges from undirected graph
   EXPECT_EQ( directedGraph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 1, 0 ), 1.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 2, 1 ), 2.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 0, 2 ), 3.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 2, 0 ), 3.0 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      SymmetricGraphType symmetricGraph( undirectedGraph );

      EXPECT_EQ( symmetricGraph.getNodeCount(), 3 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 3 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 1 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 0 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 2 ), 2.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 1 ), 2.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 2 ), 3.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 0 ), 3.0 );
   }
}

TYPED_TEST( GraphBasicTest, MoveConstructor )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 } } );

   GraphType graph2( std::move( graph1 ) );

   EXPECT_EQ( graph2.getNodeCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 3 );

   EXPECT_EQ( graph2.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 0, 2 ), 3.0 );
}

TYPED_TEST( GraphBasicTest, CopyAssignment )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 }, { 2, 0, 4.0 } } );
   GraphType graph2;

   graph2 = graph1;

   EXPECT_EQ( graph2.getNodeCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 4 );

   EXPECT_EQ( graph2.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 0, 2 ), 3.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 2, 0 ), 4.0 );
}

TYPED_TEST( GraphBasicTest, MoveAssignment )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
   GraphType graph2;

   graph2 = std::move( graph1 );

   EXPECT_EQ( graph2.getNodeCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 2 );

   EXPECT_EQ( graph2.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 1, 2 ), 2.0 );
}

TYPED_TEST( GraphBasicTest, EqualityOperator )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
   GraphType graph2( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
   GraphType graph3( 3, { { 0, 1, 1.0 } } );

   EXPECT_TRUE( graph1 == graph2 );
   EXPECT_FALSE( graph1 == graph3 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      SymmetricGraphType symmetricGraph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
      SymmetricGraphType symmetricGraph2( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
      SymmetricGraphType symmetricGraph3( 3, { { 0, 1, 1.0 } } );

      // TODO: The following does not work. if we try to construct symmetric graphs from existing directed graphs,
      // the adjacency matrix of the directed graph is copiedto the symmetric graph. In this case,
      // only the lower triangle of the matrix is considered.  We need to take into account both triangles when constructing
      // the symmetric graph from a directed graph.
      //SymmetricGraphType symmetricGraph1( graph1 );
      //SymmetricGraphType symmetricGraph2( graph2 );
      //SymmetricGraphType symmetricGraph3( graph3 );

      EXPECT_TRUE( symmetricGraph1 == symmetricGraph2 );
      EXPECT_FALSE( symmetricGraph1 == symmetricGraph3 );
   }
}

TYPED_TEST( GraphBasicTest, EmptyGraph )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph( 5, {} );

   EXPECT_EQ( graph.getNodeCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 0 );
}

TYPED_TEST( GraphBasicTest, SingleEdge )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   GraphType graph( 2, { { 0, 1, 1.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 2 );
   EXPECT_EQ( graph.getEdgeCount(), 1 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      // TODO: The following does not work for the same reason as in the EqualityOperator test.
      //SymmetricGraphType symmetricGraph( graph );
      SymmetricGraphType symmetricGraph( 2, { { 1, 0, 1.0 } } );

      EXPECT_EQ( symmetricGraph.getNodeCount(), 2 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 1 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 1 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 0 ), 1.0 );
   }
}

TYPED_TEST( GraphBasicTest, SelfLoop )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   GraphType graph( 3, { { 0, 0, 1.0 }, { 1, 2, 2.0 }, { 2, 2, 3.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 3 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 0 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 2 ), 3.0 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      // TODO: The following does not work for the same reason as in the EqualityOperator test.
      //SymmetricGraphType symmetricGraph( graph );
      SymmetricGraphType symmetricGraph( 3, { { 0, 0, 1.0 }, { 2, 1, 2.0 }, { 2, 2, 3.0 } } );
      EXPECT_EQ( symmetricGraph.getNodeCount(), 3 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 3 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 0 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 2 ), 2.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 1 ), 2.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 2 ), 3.0 );
   }
}

TYPED_TEST( GraphBasicTest, CompleteGraph )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   // Complete graph with 3 nodes (all nodes connected to all other nodes)
   GraphType graph( 3, { { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 1, 0, 1.0 }, { 1, 2, 1.0 }, { 2, 0, 1.0 }, { 2, 1, 1.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 6 );

   // Check all edges in the complete graph
   for( int i = 0; i < 3; i++ ) {
      for( int j = 0; j < 3; j++ ) {
         if( i != j ) {
            EXPECT_EQ( graph.getEdgeWeight( i, j ), 1.0 );
         }
      }
   }

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      SymmetricGraphType symmetricGraph(
         3, { { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 1, 0, 1.0 }, { 1, 2, 1.0 }, { 2, 0, 1.0 }, { 2, 1, 1.0 } } );

      // Check all edges in the complete graph
      for( int i = 0; i < 3; i++ ) {
         for( int j = 0; j < 3; j++ ) {
            if( i != j ) {
               EXPECT_EQ( symmetricGraph.getEdgeWeight( i, j ), 1.0 );
            }
         }
      }
   }
}

TYPED_TEST( GraphBasicTest, LinearChain )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   // Linear chain: 0 -> 1 -> 2 -> 3 -> 4
   GraphType graph( 5, { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 }, { 3, 4, 1.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 4 );

   // Check all edges in the linear chain
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 3 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 3, 4 ), 1.0 );

   // Check non-existing edges
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 3 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 4 ), 0.0 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;

      // Linear chain: 0 -> 1 -> 2 -> 3 -> 4
      SymmetricGraphType symmetricGraph( 5, { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 }, { 3, 4, 1.0 } } );

      EXPECT_EQ( symmetricGraph.getNodeCount(), 5 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 4 );

      // Check all edges in the linear chain
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 1 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 2 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 3 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 3, 4 ), 1.0 );

      // Check non-existing edges
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 2 ), 0.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 3 ), 0.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 4 ), 0.0 );
   }
}

TYPED_TEST( GraphBasicTest, StarGraph )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using AdjacencyMatrixType = typename GraphType::AdjacencyMatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;

   // Star graph: central node 0 connected to nodes 1, 2, 3, 4
   GraphType graph( 5, { { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 4, 1.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 4 );

   // Check edges from central node
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 3 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 4 ), 1.0 );

   // Check non-existing edges between outer nodes
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 3 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 3 ), 0.0 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      SymmetricGraphType symmetricGraph( 5, { { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 4, 1.0 } } );
      EXPECT_EQ( symmetricGraph.getNodeCount(), 5 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 4 );

      // Check edges from central node
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 1 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 2 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 3 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 4 ), 1.0 );

      // Check non-existing edges between outer nodes
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 2 ), 0.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 3 ), 0.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 3 ), 0.0 );
   }
}

// Test for undirected graphs
TYPED_TEST( GraphBasicTest, UndirectedGraphBasic )
{
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;
   using AdjacencyMatrixType = typename UndirectedGraphType::AdjacencyMatrixType;
   using ValueType = typename UndirectedGraphType::ValueType;
   using DeviceType = typename UndirectedGraphType::DeviceType;
   using IndexType = typename UndirectedGraphType::IndexType;

   // For undirected graphs with non-symmetric matrix, edges are added in both directions
   UndirectedGraphType graph( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   // Edge count should be 2 for undirected graph (each edge is counted once)
   EXPECT_EQ( graph.getEdgeCount(), 2 );

   // Check edges are symmetric for undirected graph
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 0 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 1 ), 2.0 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType =
         TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType, TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;

      SymmetricGraphType symmetricGraph( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
      EXPECT_EQ( symmetricGraph.getNodeCount(), 3 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 2 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 1 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 0 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 2 ), 2.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 1 ), 2.0 );
   }
}

TYPED_TEST( GraphBasicTest, DirectedGraphBasic )
{
   using DirectedGraphType = typename TestFixture::DirectedGraphType;

   DirectedGraphType graph( 3, { { 0, 1, 1.0 }, { 1, 0, 2.0 }, { 1, 2, 3.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 3 );
}

TYPED_TEST( GraphBasicTest, LargerGraph )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph( 10,
                    { { 0, 1, 1.0 },
                      { 1, 2, 1.0 },
                      { 2, 3, 1.0 },
                      { 3, 4, 1.0 },
                      { 4, 5, 1.0 },
                      { 5, 6, 1.0 },
                      { 6, 7, 1.0 },
                      { 7, 8, 1.0 },
                      { 8, 9, 1.0 },
                      { 0, 9, 1.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 10 );
   EXPECT_EQ( graph.getEdgeCount(), 10 );

   // Check edges in the graph
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 3 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 3, 4 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 4, 5 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 5, 6 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 6, 7 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 7, 8 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 8, 9 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 9 ), 1.0 );

   // Check some non-existing edges
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 3 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 5, 9 ), 0.0 );

   // Test undirected graph using symmetric sparse matrix
   if constexpr( TNL::Matrices::is_sparse_matrix< typename GraphType::AdjacencyMatrixType >() ) {
      using SymmetricAdjacencyMatrixType = TNL::Matrices::SparseMatrix< typename GraphType::ValueType,
                                                                        typename GraphType::DeviceType,
                                                                        typename GraphType::IndexType,
                                                                        TNL::Matrices::SymmetricMatrix >;
      using SymmetricGraphType = TNL::Graphs::Graph< SymmetricAdjacencyMatrixType, TNL::Graphs::UndirectedGraph >;
      SymmetricGraphType symmetricGraph( 10,
                                         { { 0, 1, 1.0 },
                                           { 1, 2, 1.0 },
                                           { 2, 3, 1.0 },
                                           { 3, 4, 1.0 },
                                           { 4, 5, 1.0 },
                                           { 5, 6, 1.0 },
                                           { 6, 7, 1.0 },
                                           { 7, 8, 1.0 },
                                           { 8, 9, 1.0 },
                                           { 0, 9, 1.0 } } );

      EXPECT_EQ( symmetricGraph.getNodeCount(), 10 );
      EXPECT_EQ( symmetricGraph.getEdgeCount(), 10 );

      // Check edges in the graph
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 1 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 2 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 2, 3 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 3, 4 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 4, 5 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 5, 6 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 6, 7 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 7, 8 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 8, 9 ), 1.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 9 ), 1.0 );

      // Check some non-existing edges
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 0, 2 ), 0.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 1, 3 ), 0.0 );
      EXPECT_EQ( symmetricGraph.getEdgeWeight( 5, 9 ), 0.0 );
   }
}

#include "../main.h"
