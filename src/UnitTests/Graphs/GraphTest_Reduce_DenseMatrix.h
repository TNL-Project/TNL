// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/reduce.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Vector.h>

#include <gtest/gtest.h>

// Test fixture for dense matrix graph reduction tests
template< typename Matrix >
class DenseGraphReductionTest : public ::testing::Test
{
protected:
   using AdjacencyMatrixType = Matrix;
   using ValueType = typename AdjacencyMatrixType::RealType;
   using DeviceType = typename AdjacencyMatrixType::DeviceType;
   using IndexType = typename AdjacencyMatrixType::IndexType;

   using DirectedGraphType =
      TNL::Graphs::Graph< ValueType,
                          DeviceType,
                          IndexType,
                          TNL::Graphs::DirectedGraph,
                          TNL::Algorithms::Segments::CSR,  // this parameter is ignored for dense matrices
                          AdjacencyMatrixType >;
};

// Types for which DenseGraphReductionTest is instantiated
using DenseGraphReductionTestTypes = ::testing::Types< TNL::Matrices::DenseMatrix< double, TNL::Devices::Sequential, int >,
                                                       TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int >
#if defined( __CUDACC__ )
                                                       ,
                                                       TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
                                                       ,
                                                       TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int >
#endif
                                                       >;

TYPED_TEST_SUITE( DenseGraphReductionTest, DenseGraphReductionTestTypes );

// Helper function to create a complete directed graph (4 vertices, all connected)
// Graph structure (complete graph without self-loops):
//                   0 -> 1 ( 1.0),  0 -> 2 ( 2.0),  0 -> 3 (3.0)
//   1 -> 0 ( 4.0),                  1 -> 2 ( 5.0),  1 -> 3 (6.0)
//   2 -> 0 ( 7.0),  2 -> 1 ( 8.0),                  2 -> 3 (9.0)
//   3 -> 0 (10.0),  3 -> 1 (11.0),  3 -> 2 (12.0)
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
         else {
            graph.setEdgeWeight( i, j, ValueType( 0.0 ) );  // zero weight for self-loops which are omitted in these tests
         }
      }
   }
}

// Helper functors for CUDA compatibility
template< typename IndexType >
struct MaxWeightFetch
{
   template< typename WeightType >
   __cuda_callable__
   WeightType
   operator()( IndexType sourceIdx, IndexType targetIdx, const WeightType& weight ) const
   {
      return weight;
   }
};

template< typename ValueType >
struct MaxReduction
{
   __cuda_callable__
   ValueType
   operator()( const ValueType& a, const ValueType& b ) const
   {
      return a > b ? a : b;
   }
};

template< typename ValueType, typename IndexType >
struct MaxReductionWithArgument
{
   __cuda_callable__
   ValueType
   operator()( ValueType& a, const ValueType& b, IndexType& aIdx, const IndexType& bIdx ) const
   {
      if( b > a ) {
         a = b;
         aIdx = bIdx;
      }
      return a;
   }
};

template< typename ValueType >
struct SumReduction
{
   __cuda_callable__
   ValueType
   operator()( const ValueType& a, const ValueType& b ) const
   {
      return a + b;
   }
};

template< typename IndexType >
struct EdgeCountFetch
{
   template< typename WeightType >
   __cuda_callable__
   IndexType
   operator()( IndexType sourceIdx, IndexType targetIdx, const WeightType& weight ) const
   {
      // For dense matrices, only count non-zero edges
      if( weight != 0.0 )
         return IndexType( 1 );
      return IndexType( 0 );
   }
};

template< typename ResultView, typename ValueType >
struct StoreIntoVector
{
   mutable ResultView resultView;

   // For basic range reductions (2 parameters)
   template< typename IndexType >
   __cuda_callable__
   void
   operator()( IndexType vertexIdx, const ValueType& value ) const
   {
      resultView[ vertexIdx ] = value;
   }

   // For array-based reductions (3 parameters)
   template< typename IndexType >
   __cuda_callable__
   void
   operator()( IndexType indexOfVertexIdx, IndexType vertexIdx, const ValueType& value ) const
   {
      resultView[ vertexIdx ] = value;
   }
};

template< typename ValueView, typename IndexView, typename ValueType, typename IndexType >
struct StoreIntoVectorWithArgument
{
   mutable ValueView valueView;
   mutable IndexView indexView;

   // For range-based reductions with emptySegment (5 parameters)
   __cuda_callable__
   void
   operator()( IndexType vertexIdx, IndexType localIdx, IndexType columnIdx, const ValueType& value, bool emptySegment ) const
   {
      valueView[ vertexIdx ] = value;
      if( ! emptySegment )
         indexView[ vertexIdx ] = columnIdx;
   }

   // For array-based reductions with emptySegment (6 parameters)
   __cuda_callable__
   void
   operator()( IndexType indexOfVertexIdx,
               IndexType vertexIdx,
               IndexType localIdx,
               IndexType columnIdx,
               const ValueType& value,
               bool emptySegment ) const
   {
      valueView[ vertexIdx ] = value;
      if( ! emptySegment )
         indexView[ vertexIdx ] = columnIdx;
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

TYPED_TEST( DenseGraphReductionTest, reduceAllVertices_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Find maximum edge weight for each vertex
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 4 );
   maxWeights.setValue( 0.0 );

   auto maxWeightsView = maxWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReduction< ValueType > reduction;
   StoreIntoVector< decltype( maxWeightsView ), ValueType > store{ maxWeightsView };

   TNL::Graphs::reduceAllVertices( graph, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: max(1.0, 2.0, 3.0) = 3.0
   // Vertex 1: max(4.0, 5.0, 6.0) = 6.0
   // Vertex 2: max(7.0, 8.0, 9.0) = 9.0
   // Vertex 3: max(10.0, 11.0, 12.0) = 12.0
   EXPECT_EQ( maxWeights.getElement( 0 ), 3.0 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 6.0 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 9.0 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 12.0 );
}

TYPED_TEST( DenseGraphReductionTest, reduceAllVertices_SumWeights )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Sum edge weights for each vertex
   TNL::Containers::Vector< ValueType, DeviceType > sumWeights( 4 );
   sumWeights.setValue( 0.0 );

   auto sumWeightsView = sumWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   SumReduction< ValueType > reduction;
   StoreIntoVector< decltype( sumWeightsView ), ValueType > store{ sumWeightsView };

   TNL::Graphs::reduceAllVertices( graph, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: 1.0 + 2.0 + 3.0 = 6.0
   // Vertex 1: 4.0 + 5.0 + 6.0 = 15.0
   // Vertex 2: 7.0 + 8.0 + 9.0 = 24.0
   // Vertex 3: 10.0 + 11.0 + 12.0 = 33.0
   EXPECT_EQ( sumWeights.getElement( 0 ), 6.0 );
   EXPECT_EQ( sumWeights.getElement( 1 ), 15.0 );
   EXPECT_EQ( sumWeights.getElement( 2 ), 24.0 );
   EXPECT_EQ( sumWeights.getElement( 3 ), 33.0 );
}

TYPED_TEST( DenseGraphReductionTest, reduceAllVertices_EdgeCount )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Count edges for each vertex (compute vertex degree)
   TNL::Containers::Vector< IndexType, DeviceType > degrees( 4 );
   degrees.setValue( 0 );

   auto degreesView = degrees.getView();
   EdgeCountFetch< IndexType > fetch;
   SumReduction< IndexType > reduction;
   StoreIntoVector< decltype( degreesView ), IndexType > store{ degreesView };

   TNL::Graphs::reduceAllVertices( graph, fetch, reduction, store, IndexType( 0 ) );

   // Each vertex has 3 edges (complete graph without self-loops)
   EXPECT_EQ( degrees.getElement( 0 ), 3 );
   EXPECT_EQ( degrees.getElement( 1 ), 3 );
   EXPECT_EQ( degrees.getElement( 2 ), 3 );
   EXPECT_EQ( degrees.getElement( 3 ), 3 );
}

TYPED_TEST( DenseGraphReductionTest, reduceVertices_Range_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Find maximum edge weight for vertices 1-2
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 4 );
   maxWeights.setValue( 0.0 );

   auto maxWeightsView = maxWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReduction< ValueType > reduction;
   StoreIntoVector< decltype( maxWeightsView ), ValueType > store{ maxWeightsView };

   TNL::Graphs::reduceVertices( graph, IndexType( 1 ), IndexType( 3 ), fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: not in range = 0.0
   // Vertex 1: max(4.0, 5.0, 6.0) = 6.0
   // Vertex 2: max(7.0, 8.0, 9.0) = 9.0
   // Vertex 3: not in range = 0.0
   EXPECT_EQ( maxWeights.getElement( 0 ), 0.0 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 6.0 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 9.0 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 0.0 );
}

TYPED_TEST( DenseGraphReductionTest, reduceVertices_Array_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Find maximum edge weight for vertices 0 and 3
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 4 );
   maxWeights.setValue( 0.0 );

   TNL::Containers::Vector< IndexType, DeviceType > vertexIndices( { 0, 3 } );

   auto maxWeightsView = maxWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReduction< ValueType > reduction;
   StoreIntoVector< decltype( maxWeightsView ), ValueType > store{ maxWeightsView };

   TNL::Graphs::reduceVertices( graph, vertexIndices.getView( 0, 2 ), fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: max(1.0, 2.0, 3.0) = 3.0
   // Vertex 1: not in array = 0.0
   // Vertex 2: not in array = 0.0
   // Vertex 3: max(10.0, 11.0, 12.0) = 12.0
   EXPECT_EQ( maxWeights.getElement( 0 ), 3.0 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 0.0 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 0.0 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 12.0 );
}

TYPED_TEST( DenseGraphReductionTest, reduceAllVerticesWithArgument_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Find maximum edge weight and target vertex for each vertex
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 4 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 4 );
   maxWeights.setValue( 0.0 );
   maxTargets.setValue( -1 );

   auto maxWeightsView = maxWeights.getView();
   auto maxTargetsView = maxTargets.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReductionWithArgument< ValueType, IndexType > reduction;
   StoreIntoVectorWithArgument< decltype( maxWeightsView ), decltype( maxTargetsView ), ValueType, IndexType > store{
      maxWeightsView, maxTargetsView
   };

   TNL::Graphs::reduceAllVerticesWithArgument( graph, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: max weight 3.0 to vertex 3
   // Vertex 1: max weight 6.0 to vertex 3
   // Vertex 2: max weight 9.0 to vertex 3
   // Vertex 3: max weight 12.0 to vertex 2
   EXPECT_EQ( maxWeights.getElement( 0 ), 3.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 6.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 9.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 12.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), 2 );
}

TYPED_TEST( DenseGraphReductionTest, reduceVerticesWithArgument_Range_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Find maximum edge weight and target for vertices 0-1
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 4 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 4 );
   maxWeights.setValue( 0.0 );
   maxTargets.setValue( -1 );

   auto maxWeightsView = maxWeights.getView();
   auto maxTargetsView = maxTargets.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReductionWithArgument< ValueType, IndexType > reduction;
   StoreIntoVectorWithArgument< decltype( maxWeightsView ), decltype( maxTargetsView ), ValueType, IndexType > store{
      maxWeightsView, maxTargetsView
   };

   TNL::Graphs::reduceVerticesWithArgument( graph, IndexType( 0 ), IndexType( 2 ), fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: max weight 3.0 to vertex 3
   // Vertex 1: max weight 6.0 to vertex 3
   // Vertex 2: not in range
   // Vertex 3: not in range
   EXPECT_EQ( maxWeights.getElement( 0 ), 3.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 6.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), -1 );
}

TYPED_TEST( DenseGraphReductionTest, reduceVerticesWithArgument_Array_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Find maximum edge weight for vertices 1 and 2
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 4 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 4 );
   maxWeights.setValue( 0.0 );
   maxTargets.setValue( -1 );

   TNL::Containers::Vector< IndexType, DeviceType > vertexIndices( { 1, 2 } );

   auto maxWeightsView = maxWeights.getView();
   auto maxTargetsView = maxTargets.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReductionWithArgument< ValueType, IndexType > reduction;
   StoreIntoVectorWithArgument< decltype( maxWeightsView ), decltype( maxTargetsView ), ValueType, IndexType > store{
      maxWeightsView, maxTargetsView
   };

   TNL::Graphs::reduceVerticesWithArgument( graph, vertexIndices.getView( 0, 2 ), fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: not in array
   // Vertex 1: max weight 6.0 to vertex 3
   // Vertex 2: max weight 9.0 to vertex 3
   // Vertex 3: not in array
   EXPECT_EQ( maxWeights.getElement( 0 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 6.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 9.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), -1 );
}

TYPED_TEST( DenseGraphReductionTest, reduceAllVerticesIf_EvenVertices_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Find maximum edge weight only for even-numbered vertices
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 4 );
   maxWeights.setValue( 0.0 );

   auto maxWeightsView = maxWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReduction< ValueType > reduction;
   StoreIntoVector< decltype( maxWeightsView ), ValueType > store{ maxWeightsView };
   EvenVertexCondition< IndexType > condition;

   TNL::Graphs::reduceAllVerticesIf( graph, condition, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0 (even): max(1.0, 2.0, 3.0) = 3.0
   // Vertex 1 (odd): not processed = 0.0
   // Vertex 2 (even): max(7.0, 8.0, 9.0) = 9.0
   // Vertex 3 (odd): not processed = 0.0
   EXPECT_EQ( maxWeights.getElement( 0 ), 3.0 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 0.0 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 9.0 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 0.0 );
}

TYPED_TEST( DenseGraphReductionTest, reduceVerticesIf_Range_OddVertices_SumWeights )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Sum edge weights for vertices 0-3, but only process odd-numbered ones
   TNL::Containers::Vector< ValueType, DeviceType > sumWeights( 4 );
   sumWeights.setValue( 0.0 );

   auto sumWeightsView = sumWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   SumReduction< ValueType > reduction;
   StoreIntoVector< decltype( sumWeightsView ), ValueType > store{ sumWeightsView };
   OddVertexCondition< IndexType > condition;

   TNL::Graphs::reduceVerticesIf( graph, IndexType( 0 ), IndexType( 4 ), condition, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0 (even): not processed = 0.0
   // Vertex 1 (odd): sum(4.0, 5.0, 6.0) = 15.0
   // Vertex 2 (even): not processed = 0.0
   // Vertex 3 (odd): sum(10.0, 11.0, 12.0) = 33.0
   EXPECT_EQ( sumWeights.getElement( 0 ), 0.0 );
   EXPECT_EQ( sumWeights.getElement( 1 ), 15.0 );
   EXPECT_EQ( sumWeights.getElement( 2 ), 0.0 );
   EXPECT_EQ( sumWeights.getElement( 3 ), 33.0 );
}

TYPED_TEST( DenseGraphReductionTest, reduceAllVerticesWithArgumentIf_OddVertices_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Find maximum edge weight and target for odd-numbered vertices only
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 4 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 4 );
   maxWeights.setValue( 0.0 );
   maxTargets.setValue( -1 );

   auto maxWeightsView = maxWeights.getView();
   auto maxTargetsView = maxTargets.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReductionWithArgument< ValueType, IndexType > reduction;
   StoreIntoVectorWithArgument< decltype( maxWeightsView ), decltype( maxTargetsView ), ValueType, IndexType > store{
      maxWeightsView, maxTargetsView
   };
   OddVertexCondition< IndexType > condition;

   TNL::Graphs::reduceAllVerticesWithArgumentIf( graph, condition, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0 (even): not processed
   // Vertex 1 (odd): max weight 6.0 to vertex 3
   // Vertex 2 (even): not processed
   // Vertex 3 (odd): max weight 12.0 to vertex 2
   EXPECT_EQ( maxWeights.getElement( 0 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 6.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 12.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), 2 );
}

TYPED_TEST( DenseGraphReductionTest, reduceVerticesWithArgumentIf_Range_EvenVertices_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );

   // Find maximum edge weight and target for vertices 0-3, only even ones
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 4 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 4 );
   maxWeights.setValue( 0.0 );
   maxTargets.setValue( -1 );

   auto maxWeightsView = maxWeights.getView();
   auto maxTargetsView = maxTargets.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReductionWithArgument< ValueType, IndexType > reduction;
   StoreIntoVectorWithArgument< decltype( maxWeightsView ), decltype( maxTargetsView ), ValueType, IndexType > store{
      maxWeightsView, maxTargetsView
   };
   EvenVertexCondition< IndexType > condition;

   TNL::Graphs::reduceVerticesWithArgumentIf(
      graph, IndexType( 0 ), IndexType( 4 ), condition, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0 (even): max weight 3.0 to vertex 3
   // Vertex 1 (odd): not processed
   // Vertex 2 (even): max weight 9.0 to vertex 3
   // Vertex 3 (odd): not processed
   EXPECT_EQ( maxWeights.getElement( 0 ), 3.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 9.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), -1 );
}

TYPED_TEST( DenseGraphReductionTest, reduceAllVertices_ConstGraph )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createCompleteDirectedGraph( graph );
   const GraphType& constGraph = graph;

   // Sum edge weights using const graph
   TNL::Containers::Vector< ValueType, DeviceType > sumWeights( 4 );
   sumWeights.setValue( 0.0 );

   auto sumWeightsView = sumWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   SumReduction< ValueType > reduction;
   StoreIntoVector< decltype( sumWeightsView ), ValueType > store{ sumWeightsView };

   TNL::Graphs::reduceAllVertices( constGraph, fetch, reduction, store, ValueType( 0.0 ) );

   EXPECT_EQ( sumWeights.getElement( 0 ), 6.0 );
   EXPECT_EQ( sumWeights.getElement( 1 ), 15.0 );
   EXPECT_EQ( sumWeights.getElement( 2 ), 24.0 );
   EXPECT_EQ( sumWeights.getElement( 3 ), 33.0 );
}
