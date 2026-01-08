// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/reduce.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Containers/Vector.h>

#include <gtest/gtest.h>

// Test fixture for graph reduction tests
template< typename Matrix >
class GraphReductionTest : public ::testing::Test
{
protected:
   using AdjacencyMatrixType = Matrix;
   using DirectedGraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::DirectedGraph >;
   using ValueType = typename AdjacencyMatrixType::RealType;
   using DeviceType = typename AdjacencyMatrixType::DeviceType;
   using IndexType = typename AdjacencyMatrixType::IndexType;
};

// Types for which GraphReductionTest is instantiated
using GraphReductionTestTypes = ::testing::Types< TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int >,
                                                  TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >
#if defined( __CUDACC__ )
                                                  ,
                                                  TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
                                                  ,
                                                  TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int >
#endif
                                                  >;

TYPED_TEST_SUITE( GraphReductionTest, GraphReductionTestTypes );

// Helper function to create a simple directed graph
// Graph structure:
//                        0 -> 1 (weight 1.0), 0 -> 2 (weight 2.0)
//                                                                  1 -> 3 (weight 3.0)
//                                                                  2 -> 3 (weight 4.0)
//                                                                                       3 -> 4 (weight 5.0)
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

// Helper functors for CUDA compatibility
template< typename IndexType >
struct MaxWeightFetch
{
   __cuda_callable__
   auto
   operator()( IndexType sourceIdx, IndexType targetIdx, const auto& weight ) const
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
   __cuda_callable__
   IndexType
   operator()( IndexType sourceIdx, IndexType targetIdx, const auto& weight ) const
   {
      return IndexType( 1 );
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

TYPED_TEST( GraphReductionTest, reduceAllVertices_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Find maximum edge weight for each vertex
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 5 );
   maxWeights.setValue( 0.0 );

   auto maxWeightsView = maxWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReduction< ValueType > reduction;
   StoreIntoVector< decltype( maxWeightsView ), ValueType > store{ maxWeightsView };

   TNL::Graphs::reduceAllVertices( graph, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: max(1.0, 2.0) = 2.0
   // Vertex 1: max(3.0) = 3.0
   // Vertex 2: max(4.0) = 4.0
   // Vertex 3: max(5.0) = 5.0
   // Vertex 4: no edges = 0.0
   EXPECT_EQ( maxWeights.getElement( 0 ), 2.0 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 3.0 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 4.0 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 5.0 );
   EXPECT_EQ( maxWeights.getElement( 4 ), 0.0 );
}

TYPED_TEST( GraphReductionTest, reduceAllVertices_SumWeights )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Sum edge weights for each vertex
   TNL::Containers::Vector< ValueType, DeviceType > sumWeights( 5 );
   sumWeights.setValue( 0.0 );

   auto sumWeightsView = sumWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   SumReduction< ValueType > reduction;
   StoreIntoVector< decltype( sumWeightsView ), ValueType > store{ sumWeightsView };

   TNL::Graphs::reduceAllVertices( graph, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: 1.0 + 2.0 = 3.0
   // Vertex 1: 3.0
   // Vertex 2: 4.0
   // Vertex 3: 5.0
   // Vertex 4: 0.0
   EXPECT_EQ( sumWeights.getElement( 0 ), 3.0 );
   EXPECT_EQ( sumWeights.getElement( 1 ), 3.0 );
   EXPECT_EQ( sumWeights.getElement( 2 ), 4.0 );
   EXPECT_EQ( sumWeights.getElement( 3 ), 5.0 );
   EXPECT_EQ( sumWeights.getElement( 4 ), 0.0 );
}

TYPED_TEST( GraphReductionTest, reduceAllVertices_EdgeCount )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Count edges for each vertex (compute vertex degree)
   TNL::Containers::Vector< IndexType, DeviceType > degrees( 5 );
   degrees.setValue( 0 );

   auto degreesView = degrees.getView();
   EdgeCountFetch< IndexType > fetch;
   SumReduction< IndexType > reduction;
   StoreIntoVector< decltype( degreesView ), IndexType > store{ degreesView };

   TNL::Graphs::reduceAllVertices( graph, fetch, reduction, store, IndexType( 0 ) );

   // Vertex 0: 2 edges
   // Vertex 1: 1 edge
   // Vertex 2: 1 edge
   // Vertex 3: 1 edge
   // Vertex 4: 0 edges
   EXPECT_EQ( degrees.getElement( 0 ), 2 );
   EXPECT_EQ( degrees.getElement( 1 ), 1 );
   EXPECT_EQ( degrees.getElement( 2 ), 1 );
   EXPECT_EQ( degrees.getElement( 3 ), 1 );
   EXPECT_EQ( degrees.getElement( 4 ), 0 );
}

TYPED_TEST( GraphReductionTest, reduceVertices_Range_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Find maximum edge weight for vertices 1-3
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 5 );
   maxWeights.setValue( 0.0 );

   auto maxWeightsView = maxWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReduction< ValueType > reduction;
   StoreIntoVector< decltype( maxWeightsView ), ValueType > store{ maxWeightsView };

   TNL::Graphs::reduceVertices( graph, 1, 4, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: not in range = 0.0
   // Vertex 1: max(3.0) = 3.0
   // Vertex 2: max(4.0) = 4.0
   // Vertex 3: max(5.0) = 5.0
   // Vertex 4: not in range = 0.0
   EXPECT_EQ( maxWeights.getElement( 0 ), 0.0 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 3.0 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 4.0 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 5.0 );
   EXPECT_EQ( maxWeights.getElement( 4 ), 0.0 );
}

TYPED_TEST( GraphReductionTest, reduceVertices_Array_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Find maximum edge weight for vertices 0, 2, 4
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 5 );
   maxWeights.setValue( 0.0 );

   TNL::Containers::Vector< IndexType, DeviceType > vertexIndexes( { 0, 2, 4 } );

   auto maxWeightsView = maxWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReduction< ValueType > reduction;
   StoreIntoVector< decltype( maxWeightsView ), ValueType > store{ maxWeightsView };

   TNL::Graphs::reduceVertices( graph, vertexIndexes.getView( 0, 3 ), fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: max(1.0, 2.0) = 2.0
   // Vertex 1: not in array = 0.0
   // Vertex 2: max(4.0) = 4.0
   // Vertex 3: not in array = 0.0
   // Vertex 4: no edges = 0.0
   EXPECT_EQ( maxWeights.getElement( 0 ), 2.0 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 0.0 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 4.0 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 0.0 );
   EXPECT_EQ( maxWeights.getElement( 4 ), 0.0 );
}

TYPED_TEST( GraphReductionTest, reduceAllVerticesWithArgument_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Find maximum edge weight and target vertex for each vertex
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 5 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 5 );
   maxWeights.setValue( 0.0 );
   maxTargets.setValue( -1 );

   TNL::Containers::Vector< IndexType, DeviceType > vertexIndices( { 0, 1, 3 } );

   auto maxWeightsView = maxWeights.getView();
   auto maxTargetsView = maxTargets.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReductionWithArgument< ValueType, IndexType > reduction;
   StoreIntoVectorWithArgument< decltype( maxWeightsView ), decltype( maxTargetsView ), ValueType, IndexType > store{
      maxWeightsView, maxTargetsView
   };

   TNL::Graphs::reduceVerticesWithArgument( graph, vertexIndices.getView( 0, 3 ), fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: max weight 2.0 to vertex 2
   // Vertex 1: max weight 3.0 to vertex 3
   // Vertex 2: not in array, should remain 0.0
   // Vertex 3: max weight 5.0 to vertex 4
   // Vertex 4: no edges
   EXPECT_EQ( maxWeights.getElement( 0 ), 2.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), 2 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 3.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 5.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), 4 );
   EXPECT_EQ( maxWeights.getElement( 4 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 4 ), -1 );
}

TYPED_TEST( GraphReductionTest, reduceVerticesWithArgument_Range_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Find maximum edge weight and target for vertices 0-2
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 5 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 5 );
   maxWeights.setValue( 0.0 );
   maxTargets.setValue( -1 );

   auto maxWeightsView = maxWeights.getView();
   auto maxTargetsView = maxTargets.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReductionWithArgument< ValueType, IndexType > reduction;
   StoreIntoVectorWithArgument< decltype( maxWeightsView ), decltype( maxTargetsView ), ValueType, IndexType > store{
      maxWeightsView, maxTargetsView
   };

   TNL::Graphs::reduceVerticesWithArgument( graph, IndexType( 0 ), IndexType( 3 ), fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: max weight 2.0 to vertex 2
   // Vertex 1: max weight 3.0 to vertex 3
   // Vertex 2: max weight 4.0 to vertex 3
   // Vertex 3: not in range
   // Vertex 4: not in range
   EXPECT_EQ( maxWeights.getElement( 0 ), 2.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), 2 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 3.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 4.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 4 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 4 ), -1 );
}

TYPED_TEST( GraphReductionTest, reduceVerticesWithArgument_Array_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Find maximum edge weight for vertices 1 and 3
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 5 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 5 );
   maxWeights.setValue( 0.0 );
   maxTargets.setValue( -1 );

   TNL::Containers::Vector< IndexType, DeviceType > vertexIndexes( { 1, 3 } );

   auto maxWeightsView = maxWeights.getView();
   auto maxTargetsView = maxTargets.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReductionWithArgument< ValueType, IndexType > reduction;
   StoreIntoVectorWithArgument< decltype( maxWeightsView ), decltype( maxTargetsView ), ValueType, IndexType > store{
      maxWeightsView, maxTargetsView
   };

   TNL::Graphs::reduceVerticesWithArgument( graph, vertexIndexes.getView( 0, 2 ), fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0: not in array
   // Vertex 1: max weight 3.0 to vertex 3
   // Vertex 2: not in array
   // Vertex 3: max weight 5.0 to vertex 4
   // Vertex 4: not in array
   EXPECT_EQ( maxWeights.getElement( 0 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 3.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 5.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), 4 );
   EXPECT_EQ( maxWeights.getElement( 4 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 4 ), -1 );
}

TYPED_TEST( GraphReductionTest, reduceAllVerticesIf_EvenVertices_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Find maximum edge weight only for even-numbered vertices
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 5 );
   maxWeights.setValue( 0.0 );

   auto maxWeightsView = maxWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   MaxReduction< ValueType > reduction;
   StoreIntoVector< decltype( maxWeightsView ), ValueType > store{ maxWeightsView };
   EvenVertexCondition< IndexType > condition;

   TNL::Graphs::reduceAllVerticesIf( graph, condition, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0 (even): max(1.0, 2.0) = 2.0
   // Vertex 1 (odd): not processed = 0.0
   // Vertex 2 (even): max(4.0) = 4.0
   // Vertex 3 (odd): not processed = 0.0
   // Vertex 4 (even): no edges = 0.0
   EXPECT_EQ( maxWeights.getElement( 0 ), 2.0 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 0.0 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 4.0 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 0.0 );
   EXPECT_EQ( maxWeights.getElement( 4 ), 0.0 );
}

TYPED_TEST( GraphReductionTest, reduceVerticesIf_Range_OddVertices_SumWeights )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Sum edge weights for vertices 0-3, but only process odd-numbered ones
   TNL::Containers::Vector< ValueType, DeviceType > sumWeights( 5 );
   sumWeights.setValue( 0.0 );

   auto sumWeightsView = sumWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   SumReduction< ValueType > reduction;
   StoreIntoVector< decltype( sumWeightsView ), ValueType > store{ sumWeightsView };
   OddVertexCondition< IndexType > condition;

   TNL::Graphs::reduceVerticesIf( graph, IndexType( 0 ), IndexType( 4 ), condition, fetch, reduction, store, ValueType( 0.0 ) );

   // Vertex 0 (even): not processed = 0.0
   // Vertex 1 (odd): sum(3.0) = 3.0
   // Vertex 2 (even): not processed = 0.0
   // Vertex 3 (odd): sum(5.0) = 5.0
   // Vertex 4: not in range = 0.0
   EXPECT_EQ( sumWeights.getElement( 0 ), 0.0 );
   EXPECT_EQ( sumWeights.getElement( 1 ), 3.0 );
   EXPECT_EQ( sumWeights.getElement( 2 ), 0.0 );
   EXPECT_EQ( sumWeights.getElement( 3 ), 5.0 );
   EXPECT_EQ( sumWeights.getElement( 4 ), 0.0 );
}

TYPED_TEST( GraphReductionTest, reduceAllVerticesWithArgumentIf_OddVertices_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Find maximum edge weight and target for odd-numbered vertices only
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 5 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 5 );
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
   // Vertex 1 (odd): max weight 3.0 to vertex 3
   // Vertex 2 (even): not processed
   // Vertex 3 (odd): max weight 5.0 to vertex 4
   // Vertex 4 (even): not processed
   EXPECT_EQ( maxWeights.getElement( 0 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 3.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 5.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), 4 );
   EXPECT_EQ( maxWeights.getElement( 4 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 4 ), -1 );
}

TYPED_TEST( GraphReductionTest, reduceVerticesWithArgumentIf_Range_EvenVertices_MaxWeight )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );

   // Find maximum edge weight and target for vertices 0-3, only even ones
   TNL::Containers::Vector< ValueType, DeviceType > maxWeights( 5 );
   TNL::Containers::Vector< IndexType, DeviceType > maxTargets( 5 );
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

   // Vertex 0 (even): max weight 2.0 to vertex 2
   // Vertex 1 (odd): not processed
   // Vertex 2 (even): max weight 4.0 to vertex 3
   // Vertex 3 (odd): not processed
   // Vertex 4: not in range
   EXPECT_EQ( maxWeights.getElement( 0 ), 2.0 );
   EXPECT_EQ( maxTargets.getElement( 0 ), 2 );
   EXPECT_EQ( maxWeights.getElement( 1 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 1 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 2 ), 4.0 );
   EXPECT_EQ( maxTargets.getElement( 2 ), 3 );
   EXPECT_EQ( maxWeights.getElement( 3 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 3 ), -1 );
   EXPECT_EQ( maxWeights.getElement( 4 ), 0.0 );
   EXPECT_EQ( maxTargets.getElement( 4 ), -1 );
}

TYPED_TEST( GraphReductionTest, reduceAllVertices_ConstGraph )
{
   using GraphType = typename TestFixture::DirectedGraphType;
   using ValueType = typename TestFixture::ValueType;
   using DeviceType = typename TestFixture::DeviceType;
   using IndexType = typename TestFixture::IndexType;

   GraphType graph;
   createSimpleDirectedGraph( graph );
   const GraphType& constGraph = graph;

   // Sum edge weights using const graph
   TNL::Containers::Vector< ValueType, DeviceType > sumWeights( 5 );
   sumWeights.setValue( 0.0 );

   auto sumWeightsView = sumWeights.getView();
   MaxWeightFetch< IndexType > fetch;
   SumReduction< ValueType > reduction;
   StoreIntoVector< decltype( sumWeightsView ), ValueType > store{ sumWeightsView };

   TNL::Graphs::reduceAllVertices( constGraph, fetch, reduction, store, ValueType( 0.0 ) );

   EXPECT_EQ( sumWeights.getElement( 0 ), 3.0 );
   EXPECT_EQ( sumWeights.getElement( 1 ), 3.0 );
   EXPECT_EQ( sumWeights.getElement( 2 ), 4.0 );
   EXPECT_EQ( sumWeights.getElement( 3 ), 5.0 );
   EXPECT_EQ( sumWeights.getElement( 4 ), 0.0 );
}
