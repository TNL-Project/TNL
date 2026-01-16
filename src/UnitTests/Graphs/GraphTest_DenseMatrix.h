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
   using DirectedGraphType = TNL::Graphs::Graph< typename Matrix::RealType,
                                                 typename Matrix::DeviceType,
                                                 typename Matrix::IndexType,
                                                 TNL::Graphs::DirectedGraph,
                                                 TNL::Algorithms::Segments::CSR,
                                                 AdjacencyMatrixType >;
   using UndirectedGraphType = TNL::Graphs::Graph< typename Matrix::RealType,
                                                   typename Matrix::DeviceType,
                                                   typename Matrix::IndexType,
                                                   TNL::Graphs::UndirectedGraph,
                                                   TNL::Algorithms::Segments::CSR,
                                                   AdjacencyMatrixType >;
};

// types for which GraphBasicTest is instantiated
using GraphBasicTestTypes = ::testing::Types< TNL::Matrices::DenseMatrix< double, TNL::Devices::Sequential, int >,
                                              TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int >
#if defined( __CUDACC__ )
                                              ,
                                              TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
                                              ,
                                              TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int >
#endif
                                              >;

TYPED_TEST_SUITE( GraphBasicTest, GraphBasicTestTypes );

TYPED_TEST( GraphBasicTest, DefaultConstructor )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph;
   EXPECT_EQ( graph.getVertexCount(), 0 );
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

TYPED_TEST( GraphBasicTest, SetVertexCount )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph;
   graph.setVertexCount( 5 );
   EXPECT_EQ( graph.getVertexCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 25 );

   graph.setVertexCount( 10 );
   EXPECT_EQ( graph.getVertexCount(), 10 );

   auto graph_view = graph.getView();
   EXPECT_EQ( graph_view.getVertexCount(), 10 );

   using GraphViewType = typename GraphType::ViewType;
   EXPECT_TRUE( ( std::is_same_v< GraphViewType, decltype( graph.getView() ) > ) );
}

TYPED_TEST( GraphBasicTest, ConstructorWithInitializerList )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   // clang-format off
   GraphType graph( { { 0.0, 1.0, 2.0 },
                      { 3.0, 0.0, 4.0 },
                      { 5.0, 6.0, 0.0 } } );
   // clang-format on

   EXPECT_EQ( graph.getVertexCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 9 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 0 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 0 ), 3.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 1 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 4.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 0 ), 5.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 1 ), 6.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 2 ), 0.0 );
}

TYPED_TEST( GraphBasicTest, ConstructorWithMap )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;

   std::map< std::pair< IndexType, IndexType >, ValueType > edges;
   edges[ { 0, 1 } ] = 1.0;
   edges[ { 0, 2 } ] = 2.0;
   edges[ { 1, 2 } ] = 3.0;
   edges[ { 1, 3 } ] = 4.0;
   edges[ { 2, 3 } ] = 5.0;

   GraphType graph( 4, edges );

   EXPECT_EQ( graph.getVertexCount(), 4 );
   EXPECT_EQ( graph.getEdgeCount(), 16 );

   // Check if the edges are stored correctly in the adjacency matrix
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 3.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 3 ), 4.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 3 ), 5.0 );

   // Create the same graph as undirected
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;
   UndirectedGraphType undirectedGraph( 4, edges );

   EXPECT_EQ( undirectedGraph.getVertexCount(), 4 );
   EXPECT_EQ( undirectedGraph.getEdgeCount(), 6 );

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
}

TYPED_TEST( GraphBasicTest, SetEdgesWithInitializerList )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph;
   // clang-format off
   graph.setDenseEdges( { { 0.0, 1.5, 3.5 },
                          { 2.5, 0.0, 4.5 },
                          { 5.5, 6.5, 0.0 } } );
   // clang-format on

   EXPECT_EQ( graph.getVertexCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 9 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.5 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 3.5 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 0 ), 2.5 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 4.5 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 0 ), 5.5 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 1 ), 6.5 );

   // Verify diagonal remained zero
   EXPECT_EQ( graph.getEdgeWeight( 0, 0 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 1 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 2 ), 0.0 );
}

TYPED_TEST( GraphBasicTest, SetEdgesWithMap )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;

   // Test directed graph with map variant of setEdges
   GraphType graph;
   graph.setVertexCount( 3 );

   std::map< std::pair< IndexType, IndexType >, ValueType > edges;
   edges[ { 0, 1 } ] = 1.5;
   edges[ { 1, 2 } ] = 2.5;
   edges[ { 0, 2 } ] = 3.5;

   graph.setEdges( edges );

   EXPECT_EQ( graph.getVertexCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 9 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.5 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 2.5 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 3.5 );

   // Test for undirected graph
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;
   UndirectedGraphType undirectedGraph;
   undirectedGraph.setVertexCount( 3 );
   undirectedGraph.setEdges( edges );

   EXPECT_EQ( undirectedGraph.getVertexCount(), 3 );
   EXPECT_EQ( undirectedGraph.getEdgeCount(), 3 );

   EXPECT_EQ( undirectedGraph.getEdgeWeight( 0, 1 ), 1.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 0 ), 1.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 1, 2 ), 2.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 1 ), 2.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 0, 2 ), 3.5 );
   EXPECT_EQ( undirectedGraph.getEdgeWeight( 2, 0 ), 3.5 );
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

   EXPECT_EQ( graph.getVertexCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 9 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 0 ), 3.0 );
}

TYPED_TEST( GraphBasicTest, CopyConstructor )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 } } );

   GraphType graph2( graph1 );

   EXPECT_EQ( graph2.getVertexCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 9 );
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

   EXPECT_EQ( directedGraph.getVertexCount(), 3 );
   EXPECT_EQ( directedGraph.getEdgeCount(), 9 );  // because undirected edges become directed edges in both directions

   // Check that directed graph has symmetric edges from undirected graph
   EXPECT_EQ( directedGraph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 1, 0 ), 1.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 2, 1 ), 2.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 0, 2 ), 3.0 );
   EXPECT_EQ( directedGraph.getEdgeWeight( 2, 0 ), 3.0 );
}

TYPED_TEST( GraphBasicTest, MoveConstructor )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 } } );

   GraphType graph2( std::move( graph1 ) );

   EXPECT_EQ( graph2.getVertexCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 9 );

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

   EXPECT_EQ( graph2.getVertexCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 9 );

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

   EXPECT_EQ( graph2.getVertexCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 9 );

   EXPECT_EQ( graph2.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph2.getEdgeWeight( 1, 2 ), 2.0 );
}

TYPED_TEST( GraphBasicTest, EqualityOperator )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
   GraphType graph2( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
   GraphType graph3( 3, { { 0, 1, 1.0 } } );

   EXPECT_TRUE( graph1 == graph2 );
   EXPECT_FALSE( graph1 == graph3 );
}

TYPED_TEST( GraphBasicTest, EmptyGraph )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph( 5, {} );

   EXPECT_EQ( graph.getVertexCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 25 );
}

TYPED_TEST( GraphBasicTest, SingleEdge )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph( 2, { { 0, 1, 1.0 } } );

   EXPECT_EQ( graph.getVertexCount(), 2 );
   EXPECT_EQ( graph.getEdgeCount(), 4 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
}

TYPED_TEST( GraphBasicTest, SelfLoop )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   GraphType graph( 3, { { 0, 0, 1.0 }, { 1, 2, 2.0 }, { 2, 2, 3.0 } } );

   EXPECT_EQ( graph.getVertexCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 9 );

   EXPECT_EQ( graph.getEdgeWeight( 0, 0 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 2 ), 3.0 );
}

TYPED_TEST( GraphBasicTest, CompleteGraph )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   // Complete graph with 3 nodes (all nodes connected to all other nodes)
   GraphType graph( 3, { { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 1, 0, 1.0 }, { 1, 2, 1.0 }, { 2, 0, 1.0 }, { 2, 1, 1.0 } } );

   EXPECT_EQ( graph.getVertexCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 9 );

   // Check all edges in the complete graph
   for( int i = 0; i < 3; i++ ) {
      for( int j = 0; j < 3; j++ ) {
         if( i != j ) {
            EXPECT_EQ( graph.getEdgeWeight( i, j ), 1.0 );
         }
      }
   }
}

TYPED_TEST( GraphBasicTest, LinearChain )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   // Linear chain: 0 -> 1 -> 2 -> 3 -> 4
   GraphType graph( 5, { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 }, { 3, 4, 1.0 } } );

   EXPECT_EQ( graph.getVertexCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 25 );

   // Check all edges in the linear chain
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 3 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 3, 4 ), 1.0 );

   // Check non-existing edges
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 3 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 4 ), 0.0 );
}

TYPED_TEST( GraphBasicTest, StarGraph )
{
   using GraphType = typename TestFixture::DirectedGraphType;

   // Star graph: central node 0 connected to nodes 1, 2, 3, 4
   GraphType graph( 5, { { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 4, 1.0 } } );

   EXPECT_EQ( graph.getVertexCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 25 );

   // Check edges from central node
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 2 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 3 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 0, 4 ), 1.0 );

   // Check non-existing edges between outer nodes
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 3 ), 0.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 3 ), 0.0 );
}

// Test for undirected graphs
TYPED_TEST( GraphBasicTest, UndirectedGraphBasic )
{
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;

   // For undirected graphs with non-symmetric matrix, edges are added in both directions
   UndirectedGraphType graph( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );

   EXPECT_EQ( graph.getVertexCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 3 );

   // Check edges are symmetric for undirected graph
   EXPECT_EQ( graph.getEdgeWeight( 0, 1 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 0 ), 1.0 );
   EXPECT_EQ( graph.getEdgeWeight( 1, 2 ), 2.0 );
   EXPECT_EQ( graph.getEdgeWeight( 2, 1 ), 2.0 );
}

TYPED_TEST( GraphBasicTest, DirectedGraphBasic )
{
   using DirectedGraphType = typename TestFixture::DirectedGraphType;

   DirectedGraphType graph( 3, { { 0, 1, 1.0 }, { 1, 0, 2.0 }, { 1, 2, 3.0 } } );

   EXPECT_EQ( graph.getVertexCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 9 );
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

   EXPECT_EQ( graph.getVertexCount(), 10 );
   EXPECT_EQ( graph.getEdgeCount(), 100 );

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
}

#include "../main.h"
