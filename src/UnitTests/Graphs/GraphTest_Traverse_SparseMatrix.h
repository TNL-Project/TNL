// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Vector.h>

#include <gtest/gtest.h>

// Test fixture for graph traversal tests
template< typename Matrix >
class GraphTraversalTest : public ::testing::Test
{
protected:
   using AdjacencyMatrixType = Matrix;
   using DirectedGraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::DirectedGraph >;
   using UndirectedGraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::UndirectedGraph >;
   using ValueType = typename AdjacencyMatrixType::RealType;
   using DeviceType = typename AdjacencyMatrixType::DeviceType;
   using IndexType = typename AdjacencyMatrixType::IndexType;
};

// Types for which GraphTraversalTest is instantiated
using GraphTraversalTestTypes = ::testing::Types< TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int >,
                                                  TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >
#if defined( __CUDACC__ )
                                                  ,
                                                  TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
                                                  ,
                                                  TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int >
#endif
                                                  >;

TYPED_TEST_SUITE( GraphTraversalTest, GraphTraversalTestTypes );

// Helper function to create a simple directed graph
// Graph structure:
//   0 -> 1 (weight 1.0)
//   0 -> 2 (weight 2.0)
//   1 -> 3 (weight 3.0)
//   2 -> 3 (weight 4.0)
//   3 -> 4 (weight 5.0)
template< typename GraphType >
void
createSimpleDirectedGraph( GraphType& graph )
{
   using ValueType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType >;

   graph.setVertexCount( 5 );
   graph.setVertexCapacities( IndexVector( { 2, 1, 1, 1, 0 } ) );

   // Set edges
   graph.setEdgeWeight( 0, 1, ValueType( 1.0 ) );
   graph.setEdgeWeight( 0, 2, ValueType( 2.0 ) );
   graph.setEdgeWeight( 1, 3, ValueType( 3.0 ) );
   graph.setEdgeWeight( 2, 3, ValueType( 4.0 ) );
   graph.setEdgeWeight( 3, 4, ValueType( 5.0 ) );
}

TYPED_TEST( GraphTraversalTest, forAllEdges )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Count edges using forAllEdges
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   auto edgeCountView = edgeCount.getView();
   auto func_count = [ edgeCountView ] __cuda_callable__(
                        IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, ValueType & edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( edgeCountView[ 0 ], IndexType( 1 ) );
   };

   TNL::Graphs::forAllEdges( graph, func_count );

   EXPECT_EQ( edgeCount.getElement( 0 ), 5 );

   // Sum all edge weights
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto weightSumView = weightSum.getView();
   auto func_sum = [ weightSumView ] __cuda_callable__(
                      IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, const ValueType& edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( weightSumView[ 0 ], edgeWeight );
   };
   const GraphType& constGraph = graph;
   TNL::Graphs::forAllEdges( constGraph, func_sum );

   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 15.0 );  // 1+2+3+4+5 = 15
}

TYPED_TEST( GraphTraversalTest, forEdges_Range )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Process edges only in vertices [1, 3)
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   auto edgeCountView = edgeCount.getView();
   auto func_count = [ edgeCountView ] __cuda_callable__(
                        IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, ValueType & edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( edgeCountView[ 0 ], IndexType( 1 ) );
   };

   TNL::Graphs::forEdges( graph, 1, 3, func_count );
   EXPECT_EQ( edgeCount.getElement( 0 ), 2 );  // vertex 1 has 1 edge, vertex 2 has 1 edge

   // Sum edge weights in vertices [0, 2)
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto weightSumView = weightSum.getView();
   auto func_sum = [ weightSumView ] __cuda_callable__(
                      IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, const ValueType& edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( weightSumView[ 0 ], edgeWeight );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forEdges( constGraph, 0, 2, func_sum );
   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 6.0 );  // vertex 0: 1+2 = 3, vertex 1: 3 => 3+3 = 6
}

TYPED_TEST( GraphTraversalTest, forEdges_Array )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Process edges only in vertices [0, 3]
   TNL::Containers::Vector< IndexType, DeviceType > vertices_1( { 0, 3 } );
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   auto edgeCountView = edgeCount.getView();
   auto func_count = [ edgeCountView ] __cuda_callable__(
                        IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, ValueType & edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( edgeCountView[ 0 ], IndexType( 1 ) );
   };

   TNL::Graphs::forEdges( graph, vertices_1, 0, 2, func_count );
   EXPECT_EQ( edgeCount.getElement( 0 ), 3 );  // vertex 0 has 2 edges, vertex 3 has 1 edge

   // Process edges in vertices [1, 2]
   TNL::Containers::Vector< IndexType, DeviceType > vertices_2( { 1, 2 } );
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto weightSumView = weightSum.getView();
   auto func_sum = [ weightSumView ] __cuda_callable__(
                      IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, const ValueType& edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( weightSumView[ 0 ], edgeWeight );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forEdges( constGraph, vertices_2, func_sum );
   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 7.0 );  // vertex 1: 3, vertex 2: 4 => 7
}

TYPED_TEST( GraphTraversalTest, forAllEdgesIf )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Process edges only in vertices with even indices
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   auto condition_count = [] __cuda_callable__( IndexType vertexIdx )
   {
      return vertexIdx % 2 == 0;
   };

   auto edgeCountView = edgeCount.getView();
   auto func_count = [ edgeCountView ] __cuda_callable__(
                        IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, ValueType & edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( edgeCountView[ 0 ], IndexType( 1 ) );
   };

   TNL::Graphs::forAllEdgesIf( graph, condition_count, func_count );

   EXPECT_EQ( edgeCount.getElement( 0 ), 3 );  // vertex 0: 2 edges, vertex 2: 1 edge
   // Process edges only in vertices with index > 0
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto condition_sum = [] __cuda_callable__( IndexType vertexIdx )
   {
      return vertexIdx > 0;
   };

   auto weightSumView = weightSum.getView();
   auto func_sum = [ weightSumView ] __cuda_callable__(
                      IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, const ValueType& edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( weightSumView[ 0 ], edgeWeight );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forAllEdgesIf( constGraph, condition_sum, func_sum );

   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 12.0 );  // vertices 1,2,3: 3+4+5 = 12
}

TYPED_TEST( GraphTraversalTest, forEdgesIf )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Process edges in range [1, 4) where vertex index is odd
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   auto condition_count = [] __cuda_callable__( IndexType vertexIdx )
   {
      return vertexIdx % 2 == 1;
   };

   auto edgeCountView = edgeCount.getView();
   auto func_count = [ edgeCountView ] __cuda_callable__(
                        IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, ValueType & edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( edgeCountView[ 0 ], IndexType( 1 ) );
   };

   TNL::Graphs::forEdgesIf( graph, 1, 4, condition_count, func_count );

   EXPECT_EQ( edgeCount.getElement( 0 ), 2 );  // vertex 1: 1 edge, vertex 3: 1 edge

   // Process edges in range [0, 3) where vertex has even index (=> 0,2)
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto graphView = graph.getView();
   auto condition_sum = [ graphView ] __cuda_callable__( IndexType vertexIdx )
   {
      return vertexIdx % 2 == 0;
   };

   auto weightSumView = weightSum.getView();
   auto func_sum = [ weightSumView ] __cuda_callable__(
                      IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, const ValueType& edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( weightSumView[ 0 ], edgeWeight );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forEdgesIf( constGraph, 0, 3, condition_sum, func_sum );

   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 7.0 );  // vertices 0,2: (1+2)+(4) = 7
}

TYPED_TEST( GraphTraversalTest, forAllVertices )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Count total degree using forAllVertices
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   auto totalDegreeView = totalDegree.getView();
   auto func_count = [ totalDegreeView ] __cuda_callable__( auto& vertex ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( totalDegreeView[ 0 ], vertex.getDegree() );
   };

   TNL::Graphs::forAllVertices( graph, func_count );

   EXPECT_EQ( totalDegree.getElement( 0 ), 5 );  // Total edges in graph

   // Count vertices with at least one edge
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   auto vertexCountView = vertexCount.getView();
   auto func_sum = [ vertexCountView ] __cuda_callable__( const auto& vertex ) mutable
   {
      if( vertex.getDegree() > 0 )
         TNL::Algorithms::AtomicOperations< DeviceType >::add( vertexCountView[ 0 ], IndexType( 1 ) );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forAllVertices( constGraph, func_sum );

   EXPECT_EQ( vertexCount.getElement( 0 ), 4 );  // Vertices 0,1,2,3 have edges
}

TYPED_TEST( GraphTraversalTest, forVertices_Range )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Process vertices in range [1, 4)
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   auto totalDegreeView = totalDegree.getView();
   auto func_degree = [ totalDegreeView ] __cuda_callable__( auto& vertex ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( totalDegreeView[ 0 ], vertex.getDegree() );
   };

   TNL::Graphs::forVertices( graph, 1, 4, func_degree );

   EXPECT_EQ( totalDegree.getElement( 0 ), 3 );  // Vertices 1,2,3 have 1+1+1 = 3 edges
   // Count vertices with edges in range [0, 3)
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   auto vertexCountView = vertexCount.getView();
   auto func_count = [ vertexCountView ] __cuda_callable__( const auto& vertex ) mutable
   {
      if( vertex.getDegree() > 0 )
         TNL::Algorithms::AtomicOperations< DeviceType >::add( vertexCountView[ 0 ], IndexType( 1 ) );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forVertices( constGraph, 0, 3, func_count );

   EXPECT_EQ( vertexCount.getElement( 0 ), 3 );  // Vertices 0,1,2 have edges
}

TYPED_TEST( GraphTraversalTest, forVertices_Array )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Process specific vertices [0, 2, 4]
   TNL::Containers::Vector< IndexType, DeviceType > even_vertices( { 0, 2, 4 } );
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   auto totalDegreeView = totalDegree.getView();
   auto func_degree = [ totalDegreeView ] __cuda_callable__( auto& vertex ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( totalDegreeView[ 0 ], vertex.getDegree() );
   };

   TNL::Graphs::forVertices( graph, even_vertices, func_degree );

   EXPECT_EQ( totalDegree.getElement( 0 ), 3 );  // Vertices 0,2,4 have 2+1+0 = 3 edges

   // Process specific vertices [1, 3]
   TNL::Containers::Vector< IndexType, DeviceType > odd_vertices( { 1, 3 } );
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   auto vertexCountView = vertexCount.getView();
   auto func_count = [ vertexCountView ] __cuda_callable__( const auto& vertex ) mutable
   {
      if( vertex.getDegree() > 0 )
         TNL::Algorithms::AtomicOperations< DeviceType >::add( vertexCountView[ 0 ], IndexType( 1 ) );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forVertices( constGraph, odd_vertices, 0, 2, func_count );

   EXPECT_EQ( vertexCount.getElement( 0 ), 2 );  // Both vertices 1 and 3 have edges
}

TYPED_TEST( GraphTraversalTest, forAllVerticesIf )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Process vertices with even indices
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   auto condition_degree = [] __cuda_callable__( IndexType vertexIdx )
   {
      return vertexIdx % 2 == 0;
   };

   auto totalDegreeView = totalDegree.getView();
   auto func_degree = [ totalDegreeView ] __cuda_callable__( auto& vertex ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( totalDegreeView[ 0 ], vertex.getDegree() );
   };

   TNL::Graphs::forAllVerticesIf( graph, condition_degree, func_degree );

   EXPECT_EQ( totalDegree.getElement( 0 ), 3 );  // Vertices 0,2,4 have 2+1+0 = 3 edges

   // Process vertices with index < 3
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   auto condition_count = [] __cuda_callable__( IndexType vertexIdx )
   {
      return vertexIdx < 3;
   };

   auto vertexCountView = vertexCount.getView();
   auto func_count = [ vertexCountView ] __cuda_callable__( const auto& vertex ) mutable
   {
      if( vertex.getDegree() > 0 )
         TNL::Algorithms::AtomicOperations< DeviceType >::add( vertexCountView[ 0 ], IndexType( 1 ) );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forAllVerticesIf( constGraph, condition_count, func_count );

   EXPECT_EQ( vertexCount.getElement( 0 ), 3 );  // Vertices 0,1,2 have edges
}

TYPED_TEST( GraphTraversalTest, forVerticesIf )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Process vertices in range [1, 5) where index is odd
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   auto condition_degree = [] __cuda_callable__( IndexType vertexIdx )
   {
      return vertexIdx % 2 == 1;
   };

   auto totalDegreeView = totalDegree.getView();
   auto func_degree = [ totalDegreeView ] __cuda_callable__( auto& vertex ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( totalDegreeView[ 0 ], vertex.getDegree() );
   };

   TNL::Graphs::forVerticesIf( graph, 1, 5, condition_degree, func_degree );

   EXPECT_EQ( totalDegree.getElement( 0 ), 2 );  // Vertices 1,3 have 1+1 = 2 edges

   // Process even vertices in range [0, 4) with even index (=> 0,2)
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   auto graphView = graph.getView();
   auto condition_count = [ graphView ] __cuda_callable__( IndexType vertexIdx )
   {
      return ( vertexIdx % 2 ) == 0;
   };

   auto vertexCountView = vertexCount.getView();
   auto func_count = [ vertexCountView ] __cuda_callable__( const auto& vertex ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( vertexCountView[ 0 ], IndexType( 1 ) );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forVerticesIf( constGraph, 0, 4, condition_count, func_count );

   EXPECT_EQ( vertexCount.getElement( 0 ), 2 );  // Vertices 0,2
}

TYPED_TEST( GraphTraversalTest, EdgeWeightModification )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Double all edge weights
   auto func =
      [] __cuda_callable__( IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, ValueType & edgeWeight ) mutable
   {
      edgeWeight *= 2.0;
   };

   TNL::Graphs::forAllEdges( graph, func );

   // Verify weights were doubled
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto weightSumView = weightSum.getView();
   auto sumFunc = [ weightSumView ] __cuda_callable__(
                     IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, const ValueType& edgeWeight ) mutable
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( weightSumView[ 0 ], edgeWeight );
   };

   const GraphType& constGraph = graph;
   TNL::Graphs::forAllEdges( constGraph, sumFunc );

   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 30.0 );  // 2*(1+2+3+4+5) = 30
}

#include "../main.h"
