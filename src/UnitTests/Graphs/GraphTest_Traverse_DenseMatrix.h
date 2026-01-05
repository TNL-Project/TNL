// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Vector.h>

#include <gtest/gtest.h>

// Test fixture for dense graph traversal tests
template< typename Matrix >
class DenseGraphTraversalTest : public ::testing::Test
{
protected:
   using AdjacencyMatrixType = Matrix;
   using DirectedGraphType =
      TNL::Graphs::Graph< typename Matrix::RealType,
                          typename Matrix::DeviceType,
                          typename Matrix::IndexType,
                          TNL::Graphs::DirectedGraph,
                          TNL::Algorithms::Segments::Ellpack,  // this parameter is ignored for dense matrices
                          Matrix >;
   using UndirectedGraphType =
      TNL::Graphs::Graph< typename Matrix::RealType,
                          typename Matrix::DeviceType,
                          typename Matrix::IndexType,
                          TNL::Graphs::UndirectedGraph,
                          TNL::Algorithms::Segments::Ellpack,  // this parameter is ignored for dense matrices
                          Matrix >;
   using ValueType = typename AdjacencyMatrixType::RealType;
   using DeviceType = typename AdjacencyMatrixType::DeviceType;
   using IndexType = typename AdjacencyMatrixType::IndexType;
};

// Types for which DenseGraphTraversalTest is instantiated
using DenseGraphTraversalTestTypes = ::testing::Types< TNL::Matrices::DenseMatrix< double, TNL::Devices::Sequential, int >,
                                                       TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int >
#if defined( __CUDACC__ )
                                                       ,
                                                       TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
                                                       ,
                                                       TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int >
#endif
                                                       >;

TYPED_TEST_SUITE( DenseGraphTraversalTest, DenseGraphTraversalTestTypes );

// Helper function to create a complete directed graph (dense matrix)
// For a complete graph with n vertices, every vertex connects to every other vertex
// Graph structure (4 vertices):
//   0 -> 1,2,3 (weights 1.0, 2.0, 3.0)
//   1 -> 0,2,3 (weights 4.0, 5.0, 6.0)
//   2 -> 0,1,3 (weights 7.0, 8.0, 9.0)
//   3 -> 0,1,2 (weights 10.0, 11.0, 12.0)
template< typename GraphType >
void
createCompleteDirectedGraph( GraphType& graph )
{
   using ValueType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;

   const IndexType n = 4;
   graph.setVertexCount( n );

   ValueType weight = 1.0;
   for( IndexType i = 0; i < n; i++ ) {
      for( IndexType j = 0; j < n; j++ ) {
         if( i != j ) {
            graph.setEdgeWeight( i, j, weight );
            weight += 1.0;
         }
         else
            graph.setEdgeWeight( i, j, 0.0 );  // zero weights are omitted in these tests
      }
   }
}

// Helper functors for CUDA compatibility (lambdas with __cuda_callable__ cannot be in protected methods)
template< typename ViewType, typename DeviceType, typename IndexType >
struct EdgeCountFunctor
{
   mutable ViewType edgeCountView;

   template< typename ValueType >
   __cuda_callable__
   void
   operator()( IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, ValueType& edgeWeight ) const
   {
      // Only count non-zero edges (for dense matrices, zero means no edge)
      if( edgeWeight != ValueType( 0 ) )
         TNL::Algorithms::AtomicOperations< DeviceType >::add( edgeCountView[ 0 ], IndexType( 1 ) );
   }
};

template< typename ViewType, typename DeviceType >
struct WeightSumFunctor
{
   mutable ViewType weightSumView;

   template< typename IndexType, typename ValueType >
   __cuda_callable__
   void
   operator()( IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, const ValueType& edgeWeight ) const
   {
      // Only sum non-zero edges (for dense matrices, zero means no edge)
      if( edgeWeight != ValueType( 0 ) )
         TNL::Algorithms::AtomicOperations< DeviceType >::add( weightSumView[ 0 ], edgeWeight );
   }
};

template< typename IndexType >
struct EvenVertexCondition
{
   __cuda_callable__
   bool
   operator()( IndexType vertexIdx ) const
   {
      return vertexIdx % 2 == 0;
   }
};

template< typename IndexType >
struct OddVertexCondition
{
   __cuda_callable__
   bool
   operator()( IndexType vertexIdx ) const
   {
      return vertexIdx % 2 == 1;
   }
};

template< typename IndexType >
struct GreaterThanZeroCondition
{
   __cuda_callable__
   bool
   operator()( IndexType vertexIdx ) const
   {
      return vertexIdx > 0;
   }
};

template< typename IndexType >
struct LessThanThreeCondition
{
   __cuda_callable__
   bool
   operator()( IndexType vertexIdx ) const
   {
      return vertexIdx < 3;
   }
};

template< typename ViewType, typename DeviceType, typename IndexType >
struct DegreeSumFunctor
{
   mutable ViewType totalDegreeView;

   template< typename Vertex >
   __cuda_callable__
   void
   operator()( Vertex& vertex ) const
   {
      // For dense matrices, manually count non-zero edges
      IndexType degree = 0;
      for( IndexType i = 0; i < vertex.getDegree(); i++ ) {
         if( vertex.getEdgeWeight( i ) != 0.0 )
            degree++;
      }
      TNL::Algorithms::AtomicOperations< DeviceType >::add( totalDegreeView[ 0 ], degree );
   }
};

template< typename ViewType, typename DeviceType, typename IndexType >
struct VertexCountFunctor
{
   mutable ViewType vertexCountView;

   template< typename Vertex >
   __cuda_callable__
   void
   operator()( const Vertex& vertex ) const
   {
      // For dense matrices, only count vertices with at least one non-zero edge
      for( IndexType i = 0; i < vertex.getDegree(); i++ ) {
         if( vertex.getEdgeWeight( i ) != 0.0 ) {
            TNL::Algorithms::AtomicOperations< DeviceType >::add( vertexCountView[ 0 ], IndexType( 1 ) );
            return;
         }
      }
   }
};

template< typename ViewType, typename DeviceType, typename IndexType >
struct SimpleVertexCountFunctor
{
   mutable ViewType vertexCountView;

   template< typename Vertex >
   __cuda_callable__
   void
   operator()( const Vertex& vertex ) const
   {
      TNL::Algorithms::AtomicOperations< DeviceType >::add( vertexCountView[ 0 ], IndexType( 1 ) );
   }
};

struct DoubleWeightFunctor
{
   template< typename IndexType, typename ValueType >
   __cuda_callable__
   void
   operator()( IndexType vertexIdx, IndexType localIdx, IndexType neighborIdx, ValueType& edgeWeight ) const
   {
      edgeWeight *= 2.0;
   }
};

TYPED_TEST( DenseGraphTraversalTest, forAllEdges )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Count edges using forAllEdges - complete graph with 4 vertices has 4*3 = 12 edges
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   auto edgeCountView = edgeCount.getView();
   EdgeCountFunctor< decltype( edgeCountView ), DeviceType, IndexType > func_count{ edgeCountView };

   TNL::Graphs::forAllEdges( graph, func_count );

   EXPECT_EQ( edgeCount.getElement( 0 ), 12 );

   // Sum all edge weights (1+2+3+4+5+6+7+8+9+10+11+12 = 78)
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto weightSumView = weightSum.getView();
   WeightSumFunctor< decltype( weightSumView ), DeviceType > func_sum{ weightSumView };
   const GraphType& constGraph = graph;
   TNL::Graphs::forAllEdges( constGraph, func_sum );

   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 78.0 );
}

TYPED_TEST( DenseGraphTraversalTest, forEdges_Range )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Process edges only in vertices [1, 3) - vertices 1 and 2, each has 3 edges
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   auto edgeCountView = edgeCount.getView();
   EdgeCountFunctor< decltype( edgeCountView ), DeviceType, IndexType > func_count{ edgeCountView };

   TNL::Graphs::forEdges( graph, 1, 3, func_count );
   EXPECT_EQ( edgeCount.getElement( 0 ), 6 );  // vertices 1,2 each have 3 edges

   // Sum edge weights in vertices [0, 2) - vertices 0,1
   // Vertex 0: 1+2+3 = 6, Vertex 1: 4+5+6 = 15, Total = 21
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto weightSumView = weightSum.getView();
   WeightSumFunctor< decltype( weightSumView ), DeviceType > func_sum{ weightSumView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forEdges( constGraph, 0, 2, func_sum );
   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 21.0 );
}

TYPED_TEST( DenseGraphTraversalTest, forEdges_Array )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Process edges only in vertices [0, 3] - vertices 0 and 3
   TNL::Containers::Vector< IndexType, DeviceType > vertices_1( { 0, 3 } );
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   auto edgeCountView = edgeCount.getView();
   EdgeCountFunctor< decltype( edgeCountView ), DeviceType, IndexType > func_count{ edgeCountView };

   TNL::Graphs::forEdges( graph, vertices_1, 0, 2, func_count );
   EXPECT_EQ( edgeCount.getElement( 0 ), 6 );  // vertices 0,3 each have 3 edges

   // Process edges in vertices [1, 2]
   // Vertex 1: 4+5+6 = 15, Vertex 2: 7+8+9 = 24, Total = 39
   TNL::Containers::Vector< IndexType, DeviceType > vertices_2( { 1, 2 } );
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto weightSumView = weightSum.getView();
   WeightSumFunctor< decltype( weightSumView ), DeviceType > func_sum{ weightSumView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forEdges( constGraph, vertices_2, func_sum );
   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 39.0 );
}

TYPED_TEST( DenseGraphTraversalTest, forAllEdgesIf )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Process edges only in vertices with even indices (0, 2) - 6 edges total
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   EvenVertexCondition< IndexType > condition_count;

   auto edgeCountView = edgeCount.getView();
   EdgeCountFunctor< decltype( edgeCountView ), DeviceType, IndexType > func_count{ edgeCountView };

   TNL::Graphs::forAllEdgesIf( graph, condition_count, func_count );

   EXPECT_EQ( edgeCount.getElement( 0 ), 6 );  // vertices 0,2 each have 3 edges
   // Process edges only in vertices with index > 0 (vertices 1,2,3) - 9 edges
   // Weights: vertices 1,2,3 = (4+5+6)+(7+8+9)+(10+11+12) = 15+24+33 = 72
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   GreaterThanZeroCondition< IndexType > condition_sum;

   auto weightSumView = weightSum.getView();
   WeightSumFunctor< decltype( weightSumView ), DeviceType > func_sum{ weightSumView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forAllEdgesIf( constGraph, condition_sum, func_sum );

   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 72.0 );
}

TYPED_TEST( DenseGraphTraversalTest, forEdgesIf )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Process edges in range [1, 4) where vertex index is odd (vertices 1,3) - 6 edges
   TNL::Containers::Vector< IndexType, DeviceType > edgeCount( 1 );
   edgeCount.setValue( 0 );

   OddVertexCondition< IndexType > condition_count;

   auto edgeCountView = edgeCount.getView();
   EdgeCountFunctor< decltype( edgeCountView ), DeviceType, IndexType > func_count{ edgeCountView };

   TNL::Graphs::forEdgesIf( graph, 1, 4, condition_count, func_count );

   EXPECT_EQ( edgeCount.getElement( 0 ), 6 );  // vertices 1,3 each have 3 edges

   // Process even vertices in range [0, 4) with even index (vertices 0,2) - 6 edges
   // Weights: vertex 0: 1+2+3 = 6, vertex 2: 7+8+9 = 24, Total = 30
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   EvenVertexCondition< IndexType > condition_sum;

   auto weightSumView = weightSum.getView();
   WeightSumFunctor< decltype( weightSumView ), DeviceType > func_sum{ weightSumView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forEdgesIf( constGraph, 0, 4, condition_sum, func_sum );

   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 30.0 );
}

TYPED_TEST( DenseGraphTraversalTest, forAllVertices )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Count total degree using forAllVertices - each of 4 vertices has degree 3
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   auto totalDegreeView = totalDegree.getView();
   DegreeSumFunctor< decltype( totalDegreeView ), DeviceType, IndexType > func_count{ totalDegreeView };

   TNL::Graphs::forAllVertices( graph, func_count );

   EXPECT_EQ( totalDegree.getElement( 0 ), 12 );  // 4 vertices * 3 edges each

   // Count all vertices (all have edges in complete graph)
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   auto vertexCountView = vertexCount.getView();
   VertexCountFunctor< decltype( vertexCountView ), DeviceType, IndexType > func_sum{ vertexCountView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forAllVertices( constGraph, func_sum );

   EXPECT_EQ( vertexCount.getElement( 0 ), 4 );  // All 4 vertices have edges
}

TYPED_TEST( DenseGraphTraversalTest, forVertices_Range )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Process vertices in range [1, 4) - vertices 1,2,3 each with degree 3
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   auto totalDegreeView = totalDegree.getView();
   DegreeSumFunctor< decltype( totalDegreeView ), DeviceType, IndexType > func_degree{ totalDegreeView };

   TNL::Graphs::forVertices( graph, 1, 4, func_degree );

   EXPECT_EQ( totalDegree.getElement( 0 ), 9 );  // Vertices 1,2,3 have 3+3+3 = 9 edges
   // Count vertices with edges in range [0, 3)
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   auto vertexCountView = vertexCount.getView();
   VertexCountFunctor< decltype( vertexCountView ), DeviceType, IndexType > func_count{ vertexCountView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forVertices( constGraph, 0, 3, func_count );

   EXPECT_EQ( vertexCount.getElement( 0 ), 3 );  // Vertices 0,1,2 all have edges
}

TYPED_TEST( DenseGraphTraversalTest, forVertices_Array )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Process even-indexed vertices [0, 2]
   TNL::Containers::Vector< IndexType, DeviceType > even_vertices( { 0, 2 } );
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   auto totalDegreeView = totalDegree.getView();
   DegreeSumFunctor< decltype( totalDegreeView ), DeviceType, IndexType > func_degree{ totalDegreeView };

   TNL::Graphs::forVertices( graph, even_vertices, func_degree );

   EXPECT_EQ( totalDegree.getElement( 0 ), 6 );  // Vertices 0,2 each have 3 edges

   // Count odd-indexed vertices in range [0, 2)
   TNL::Containers::Vector< IndexType, DeviceType > odd_vertices( { 1, 3 } );
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   auto vertexCountView = vertexCount.getView();
   VertexCountFunctor< decltype( vertexCountView ), DeviceType, IndexType > func_count{ vertexCountView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forVertices( constGraph, odd_vertices, 0, 2, func_count );

   EXPECT_EQ( vertexCount.getElement( 0 ), 2 );  // Both vertices 1 and 3 have edges
}

TYPED_TEST( DenseGraphTraversalTest, forAllVerticesIf )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Process vertices with even indices (0,2)
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   EvenVertexCondition< IndexType > condition_degree;

   auto totalDegreeView = totalDegree.getView();
   DegreeSumFunctor< decltype( totalDegreeView ), DeviceType, IndexType > func_degree{ totalDegreeView };

   TNL::Graphs::forAllVerticesIf( graph, condition_degree, func_degree );

   EXPECT_EQ( totalDegree.getElement( 0 ), 6 );  // Vertices 0,2 have 3+3 = 6 edges

   // Process vertices with index < 3 (vertices 0,1,2)
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   LessThanThreeCondition< IndexType > condition_count;

   auto vertexCountView = vertexCount.getView();
   VertexCountFunctor< decltype( vertexCountView ), DeviceType, IndexType > func_count{ vertexCountView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forAllVerticesIf( constGraph, condition_count, func_count );

   EXPECT_EQ( vertexCount.getElement( 0 ), 3 );  // Vertices 0,1,2 have edges
}

TYPED_TEST( DenseGraphTraversalTest, forVerticesIf )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Process vertices in range [1, 4) where index is odd (vertices 1,3)
   TNL::Containers::Vector< IndexType, DeviceType > totalDegree( 1 );
   totalDegree.setValue( 0 );

   OddVertexCondition< IndexType > condition_degree;

   auto totalDegreeView = totalDegree.getView();
   DegreeSumFunctor< decltype( totalDegreeView ), DeviceType, IndexType > func_degree{ totalDegreeView };

   TNL::Graphs::forVerticesIf( graph, 1, 4, condition_degree, func_degree );

   EXPECT_EQ( totalDegree.getElement( 0 ), 6 );  // Vertices 1,3 have 3+3 = 6 edges

   // Process even vertices in range [0, 4) with even index (vertices 0,2)
   TNL::Containers::Vector< IndexType, DeviceType > vertexCount( 1 );
   vertexCount.setValue( 0 );

   EvenVertexCondition< IndexType > condition_count;

   auto vertexCountView = vertexCount.getView();
   SimpleVertexCountFunctor< decltype( vertexCountView ), DeviceType, IndexType > func_count{ vertexCountView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forVerticesIf( constGraph, 0, 4, condition_count, func_count );

   EXPECT_EQ( vertexCount.getElement( 0 ), 2 );  // Vertices 0,2
}

TYPED_TEST( DenseGraphTraversalTest, EdgeWeightModification )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Double all edge weights
   DoubleWeightFunctor func;

   TNL::Graphs::forAllEdges( graph, func );

   // Verify weights were doubled (original sum was 78, doubled is 156)
   TNL::Containers::Vector< ValueType, DeviceType > weightSum( 1 );
   weightSum.setValue( 0.0 );

   auto weightSumView = weightSum.getView();
   WeightSumFunctor< decltype( weightSumView ), DeviceType > sumFunc{ weightSumView };

   const GraphType& constGraph = graph;
   TNL::Graphs::forAllEdges( constGraph, sumFunc );

   EXPECT_DOUBLE_EQ( weightSum.getElement( 0 ), 156.0 );  // 2*78 = 156
}

#include "../main.h"
