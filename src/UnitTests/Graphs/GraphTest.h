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
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::Graph< MatrixType >;
   using DirectedGraphType = TNL::Graphs::Graph< MatrixType, TNL::Graphs::DirectedGraph >;
   using UndirectedGraphType = TNL::Graphs::Graph< MatrixType, TNL::Graphs::UndirectedGraph >;
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
   using GraphType = typename TestFixture::GraphType;

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
   using GraphType = typename TestFixture::GraphType;

   GraphType graph;
   graph.setNodeCount( 5 );
   EXPECT_EQ( graph.getNodeCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 0 );

   graph.setNodeCount( 10 );
   EXPECT_EQ( graph.getNodeCount(), 10 );
}

TYPED_TEST( GraphBasicTest, ConstructorWithInitializerList )
{
   using GraphType = typename TestFixture::GraphType;

   // Create a simple directed graph with 4 nodes and 5 edges
   GraphType graph( 4, { { 0, 1, 1.0 }, { 0, 2, 2.0 }, { 1, 2, 3.0 }, { 1, 3, 4.0 }, { 2, 3, 5.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 4 );
   EXPECT_EQ( graph.getEdgeCount(), 5 );

   // Check if the edges are stored correctly in the adjacency matrix
   const auto& matrix = graph.getAdjacencyMatrix();
   EXPECT_EQ( matrix.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( matrix.getElement( 0, 2 ), 2.0 );
   EXPECT_EQ( matrix.getElement( 1, 2 ), 3.0 );
   EXPECT_EQ( matrix.getElement( 1, 3 ), 4.0 );
   EXPECT_EQ( matrix.getElement( 2, 3 ), 5.0 );

   // Verify that non-existing edges have value 0
   EXPECT_EQ( matrix.getElement( 1, 0 ), 0.0 );
   EXPECT_EQ( matrix.getElement( 2, 0 ), 0.0 );
   EXPECT_EQ( matrix.getElement( 3, 0 ), 0.0 );

   // Create the same graph as undirected
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;
   UndirectedGraphType undirectedGraph( 4, { { 0, 1, 1.0 }, { 0, 2, 2.0 }, { 1, 2, 3.0 }, { 1, 3, 4.0 }, { 2, 3, 5.0 } } );

   EXPECT_EQ( undirectedGraph.getNodeCount(), 4 );
   EXPECT_EQ( undirectedGraph.getEdgeCount(), 5 );

   // Check if the edges are symmetric in the adjacency matrix for undirected graph
   const auto& undirectedMatrix = undirectedGraph.getAdjacencyMatrix();
   EXPECT_EQ( undirectedMatrix.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 1, 0 ), 1.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 0, 2 ), 2.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 2, 0 ), 2.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 1, 2 ), 3.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 2, 1 ), 3.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 1, 3 ), 4.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 3, 1 ), 4.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 2, 3 ), 5.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 3, 2 ), 5.0 );
}

TYPED_TEST( GraphBasicTest, ConstructorWithMap )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;

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
   const auto& matrix = graph.getAdjacencyMatrix();
   EXPECT_EQ( matrix.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( matrix.getElement( 0, 2 ), 2.0 );
   EXPECT_EQ( matrix.getElement( 1, 2 ), 3.0 );
   EXPECT_EQ( matrix.getElement( 1, 3 ), 4.0 );
   EXPECT_EQ( matrix.getElement( 2, 3 ), 5.0 );

   // Create the same graph as undirected
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;
   UndirectedGraphType undirectedGraph( 4, edges );

   EXPECT_EQ( undirectedGraph.getNodeCount(), 4 );
   EXPECT_EQ( undirectedGraph.getEdgeCount(), 5 );

   // Check if the edges are symmetric in the adjacency matrix for undirected graph
   const auto& undirectedMatrix = undirectedGraph.getAdjacencyMatrix();
   EXPECT_EQ( undirectedMatrix.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 1, 0 ), 1.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 0, 2 ), 2.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 2, 0 ), 2.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 1, 2 ), 3.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 2, 1 ), 3.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 1, 3 ), 4.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 3, 1 ), 4.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 2, 3 ), 5.0 );
   EXPECT_EQ( undirectedMatrix.getElement( 3, 2 ), 5.0 );
}

TYPED_TEST( GraphBasicTest, SetEdges )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;

   GraphType graph;
   graph.setNodeCount( 3 );

   std::map< std::pair< IndexType, IndexType >, ValueType > edges;
   edges[ { 0, 1 } ] = 1.5;
   edges[ { 1, 2 } ] = 2.5;
   edges[ { 0, 2 } ] = 3.5;

   graph.setEdges( edges );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 3 );

   const auto& matrix = graph.getAdjacencyMatrix();
   EXPECT_EQ( matrix.getElement( 0, 1 ), 1.5 );
   EXPECT_EQ( matrix.getElement( 1, 2 ), 2.5 );
   EXPECT_EQ( matrix.getElement( 0, 2 ), 3.5 );

   // Test for undirected graph
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;
   UndirectedGraphType undirectedGraph;
   undirectedGraph.setNodeCount( 3 );
   undirectedGraph.setEdges( edges );

   EXPECT_EQ( undirectedGraph.getNodeCount(), 3 );
   EXPECT_EQ( undirectedGraph.getEdgeCount(), 3 );

   const auto& undirectedMatrix = undirectedGraph.getAdjacencyMatrix();
   EXPECT_EQ( undirectedMatrix.getElement( 0, 1 ), 1.5 );
   EXPECT_EQ( undirectedMatrix.getElement( 1, 0 ), 1.5 );
   EXPECT_EQ( undirectedMatrix.getElement( 1, 2 ), 2.5 );
   EXPECT_EQ( undirectedMatrix.getElement( 2, 1 ), 2.5 );
   EXPECT_EQ( undirectedMatrix.getElement( 0, 2 ), 3.5 );
   EXPECT_EQ( undirectedMatrix.getElement( 2, 0 ), 3.5 );
}

TYPED_TEST( GraphBasicTest, GetAdjacencyMatrix )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );

   const auto& matrix = graph.getAdjacencyMatrix();
   EXPECT_EQ( matrix.getRows(), 3 );
   EXPECT_EQ( matrix.getColumns(), 3 );
   EXPECT_EQ( matrix.getNonzeroElementsCount(), 2 );
}

TYPED_TEST( GraphBasicTest, SetAdjacencyMatrix )
{
   using GraphType = typename TestFixture::GraphType;
   using MatrixType = typename GraphType::MatrixType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;

   MatrixType matrix;
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

   const auto& adjacencyMatrix = graph.getAdjacencyMatrix();
   EXPECT_EQ( adjacencyMatrix.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( adjacencyMatrix.getElement( 1, 2 ), 2.0 );
   EXPECT_EQ( adjacencyMatrix.getElement( 2, 0 ), 3.0 );
}

TYPED_TEST( GraphBasicTest, CopyConstructor )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 } } );

   GraphType graph2( graph1 );

   EXPECT_EQ( graph2.getNodeCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 3 );
   EXPECT_EQ( graph1, graph2 );

   const auto& matrix1 = graph1.getAdjacencyMatrix();
   const auto& matrix2 = graph2.getAdjacencyMatrix();
   EXPECT_EQ( matrix1.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( matrix1.getElement( 1, 2 ), 2.0 );
   EXPECT_EQ( matrix1.getElement( 0, 2 ), 3.0 );
   EXPECT_EQ( matrix2.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( matrix2.getElement( 1, 2 ), 2.0 );
   EXPECT_EQ( matrix2.getElement( 0, 2 ), 3.0 );

   // Test assignment between directed and undirected graphs
   using DirectedGraphType = typename TestFixture::DirectedGraphType;
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;

   UndirectedGraphType undirectedGraph( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 } } );
   DirectedGraphType directedGraph;

   directedGraph = undirectedGraph;

   EXPECT_EQ( directedGraph.getNodeCount(), 3 );
   EXPECT_EQ( directedGraph.getEdgeCount(), 6 );  // because undirected edges become directed edges in both directions

   const auto& directedMatrix = directedGraph.getAdjacencyMatrix();
   // Check that directed graph has symmetric edges from undirected graph
   EXPECT_EQ( directedMatrix.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( directedMatrix.getElement( 1, 0 ), 1.0 );
   EXPECT_EQ( directedMatrix.getElement( 1, 2 ), 2.0 );
   EXPECT_EQ( directedMatrix.getElement( 2, 1 ), 2.0 );
   EXPECT_EQ( directedMatrix.getElement( 0, 2 ), 3.0 );
   EXPECT_EQ( directedMatrix.getElement( 2, 0 ), 3.0 );
}

TYPED_TEST( GraphBasicTest, MoveConstructor )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 } } );

   GraphType graph2( std::move( graph1 ) );

   EXPECT_EQ( graph2.getNodeCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 3 );

   const auto& matrix = graph2.getAdjacencyMatrix();
   EXPECT_EQ( matrix.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( matrix.getElement( 1, 2 ), 2.0 );
   EXPECT_EQ( matrix.getElement( 0, 2 ), 3.0 );
}

TYPED_TEST( GraphBasicTest, CopyAssignment )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 }, { 0, 2, 3.0 }, { 2, 0, 4.0 } } );
   GraphType graph2;

   graph2 = graph1;

   EXPECT_EQ( graph2.getNodeCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 4 );

   const auto& matrix = graph2.getAdjacencyMatrix();
   EXPECT_EQ( matrix.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( matrix.getElement( 1, 2 ), 2.0 );
   EXPECT_EQ( matrix.getElement( 0, 2 ), 3.0 );
   EXPECT_EQ( matrix.getElement( 2, 0 ), 4.0 );
}

TYPED_TEST( GraphBasicTest, MoveAssignment )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
   GraphType graph2;

   graph2 = std::move( graph1 );

   EXPECT_EQ( graph2.getNodeCount(), 3 );
   EXPECT_EQ( graph2.getEdgeCount(), 2 );

   const auto& matrix = graph2.getAdjacencyMatrix();
   EXPECT_EQ( matrix.getElement( 0, 1 ), 1.0 );
   EXPECT_EQ( matrix.getElement( 1, 2 ), 2.0 );
}

TYPED_TEST( GraphBasicTest, EqualityOperator )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph1( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
   GraphType graph2( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );
   GraphType graph3( 3, { { 0, 1, 1.0 } } );

   EXPECT_TRUE( graph1 == graph2 );
   EXPECT_FALSE( graph1 == graph3 );
}

TYPED_TEST( GraphBasicTest, EmptyGraph )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph( 5, {} );

   EXPECT_EQ( graph.getNodeCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 0 );
}

TYPED_TEST( GraphBasicTest, SingleEdge )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph( 2, { { 0, 1, 1.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 2 );
   EXPECT_EQ( graph.getEdgeCount(), 1 );

   const auto& matrix = graph.getAdjacencyMatrix();
   EXPECT_EQ( matrix.getElement( 0, 1 ), 1.0 );
}

TYPED_TEST( GraphBasicTest, SelfLoop )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph( 3, { { 0, 0, 1.0 }, { 1, 2, 2.0 }, { 2, 2, 3.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 3 );
}

TYPED_TEST( GraphBasicTest, CompleteGraph )
{
   using GraphType = typename TestFixture::GraphType;

   // Complete graph with 3 nodes (all nodes connected to all other nodes)
   GraphType graph( 3, { { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 1, 0, 1.0 }, { 1, 2, 1.0 }, { 2, 0, 1.0 }, { 2, 1, 1.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 6 );
}

TYPED_TEST( GraphBasicTest, LinearChain )
{
   using GraphType = typename TestFixture::GraphType;

   // Linear chain: 0 -> 1 -> 2 -> 3 -> 4
   GraphType graph( 5, { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 }, { 3, 4, 1.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 4 );
}

TYPED_TEST( GraphBasicTest, StarGraph )
{
   using GraphType = typename TestFixture::GraphType;

   // Star graph: central node 0 connected to nodes 1, 2, 3, 4
   GraphType graph( 5, { { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 4, 1.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 5 );
   EXPECT_EQ( graph.getEdgeCount(), 4 );
}

TYPED_TEST( GraphBasicTest, SetNodeCapacities )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph;
   graph.setNodeCount( 4 );

   VectorType capacities( 4 );
   capacities.setValue( 2 );

   graph.setNodeCapacities( capacities );

   EXPECT_EQ( graph.getNodeCount(), 4 );
}

TYPED_TEST( GraphBasicTest, DifferentEdgeWeights )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph( 3, { { 0, 1, 0.5 }, { 1, 2, 1.5 }, { 0, 2, 2.5 } } );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   EXPECT_EQ( graph.getEdgeCount(), 3 );
}

// Test for undirected graphs
TYPED_TEST( GraphBasicTest, UndirectedGraphBasic )
{
   using UndirectedGraphType = typename TestFixture::UndirectedGraphType;

   // For undirected graphs with non-symmetric matrix, edges are added in both directions
   UndirectedGraphType graph( 3, { { 0, 1, 1.0 }, { 1, 2, 2.0 } } );

   EXPECT_EQ( graph.getNodeCount(), 3 );
   // Edge count should be 2 for undirected graph (each edge is counted once)
   EXPECT_EQ( graph.getEdgeCount(), 2 );
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
   using GraphType = typename TestFixture::GraphType;

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
}

#include "../main.h"
