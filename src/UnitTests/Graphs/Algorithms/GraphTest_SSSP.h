#pragma once

#include <TNL/Graphs/Algorithms/singleSourceShortestPath.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <limits>
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
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Sequential, int >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int >
#elif defined( __CUDACC__ )
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int >
#endif
   >;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

TYPED_TEST( GraphTest, test_SSSP_empty )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   GraphType graph;
   VectorType distances;
   TNL::Graphs::Algorithms::singleSourceShortestPath( graph, 0, distances );
   EXPECT_EQ( distances.getSize(), 0 );
}

TYPED_TEST( GraphTest, test_SSSP_small )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   // Create a sample graph.
   // clang-format off
   GraphType graph(
        5, // graph nodes count
        {  // definition of graph edges
                        { 0, 1, 0.5 }, { 0, 2, 1.2 },
         { 1, 0, 0.5 },                               { 1, 3, 2.3 }, { 1, 4, 3.7 },
         { 2, 0, 1.2 },                               { 2, 3, 0.8 },
                        { 3, 1, 2.3 }, { 3, 2, 0.8 },                { 3, 4, 1.5 },
                        { 4, 1, 3.7 },                { 4, 3, 1.5 }
        });
   // clang-format on

   VectorType distances( graph.getVertexCount() );
   std::vector< VectorType > expectedDistances{ { 0.0, 0.5, 1.2, 2.0, 3.5 },
                                                { 0.5, 0.0, 1.7, 2.3, 3.7 },
                                                { 1.2, 1.7, 0.0, 0.8, 2.3 },
                                                { 2.0, 2.3, 0.8, 0.0, 1.5 },
                                                { 3.5, 3.7, 2.3, 1.5, 0.0 } };

   for( int start_node = 0; start_node < graph.getVertexCount(); ++start_node ) {
      TNL::Graphs::Algorithms::singleSourceShortestPath( graph, start_node, distances );
      ASSERT_EQ( distances, expectedDistances[ start_node ] ) << "start_node: " << start_node;
   }
}

TYPED_TEST( GraphTest, test_SSSP_larger )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   // Create a larger sample graph.
   // clang-format off
   GraphType graph(
        10, // graph nodes count
        {   // definition of graph edges
                     { 0, 1, 0.5 }, { 0, 2, 1.2 },
                                                   { 1, 3, 2.3 }, { 1, 4, 3.7 },
                                                   { 2, 3, 0.8 },                { 2, 5, 2.1 },
                                                                                                { 3, 6, 1.5 },
                     { 4, 1, 3.7 },                { 4, 3, 1.5 },                                              { 4, 7, 0.9 },
                                                                                                { 5, 6, 2.4 },                { 5, 8, 1.7 },
                                                                                                                                             { 6, 9, 3.3 },
                                                                  { 7, 4, 0.4 },                                                             { 7, 9, 2.2 },
                                                                                                                                             { 8, 9, 1.9 },
                                                   { 9, 3, 0.7 }
        });
   // clang-format on

   VectorType distances( graph.getVertexCount() );
   std::vector< VectorType > expectedDistances = {
      { 0.0, 0.5, 1.2, 2.0, 4.2, 3.3, 3.5, 5.1, 5.0, 6.8 },        { -1.0, 0.0, -1.0, 2.3, 3.7, -1.0, 3.8, 4.6, -1.0, 6.8 },
      { -1.0, -1.0, 0.0, 0.8, -1.0, 2.1, 2.3, -1.0, 3.8, 5.6 },    { -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, 1.5, -1.0, -1.0, 4.8 },
      { -1.0, 3.7, -1.0, 1.5, 0.0, -1.0, 3.0, 0.9, -1.0, 3.1 },    { -1.0, -1.0, -1.0, 4.3, -1.0, 0.0, 2.4, -1.0, 1.7, 3.6 },
      { -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 0.0, -1.0, -1.0, 3.3 }, { -1.0, 4.1, -1.0, 1.9, 0.4, -1.0, 3.4, 0.0, -1.0, 2.2 },
      { -1.0, -1.0, -1.0, 2.6, -1.0, -1.0, 4.1, -1.0, 0.0, 1.9 },  { -1.0, -1.0, -1.0, 0.7, -1.0, -1.0, 2.2, -1.0, -1.0, 0.0 }
   };

   for( int start_node = 0; start_node < graph.getVertexCount(); start_node++ ) {
      TNL::Graphs::Algorithms::singleSourceShortestPath( graph, start_node, distances );
      for( IndexType i = 0; i < graph.getVertexCount(); i++ )
         ASSERT_FLOAT_EQ( distances.getElement( i ), expectedDistances[ start_node ].getElement( i ) )
            << "start_node: " << start_node << " distances[ " << i << " ]: " << distances.getElement( i )
            << " expectedDistances[ " << start_node << " ][ " << i << " ]: " << expectedDistances[ start_node ].getElement( i )
            << " distances: " << distances << " expectedDistances[ " << start_node << " ]: " << expectedDistances[ start_node ];
   }
}

TYPED_TEST( GraphTest, test_SSSP_largest )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   // Create a sample graph with 15 nodes.
   GraphType graph(
      15,  // graph nodes count
      {    // definition of graph edges
        { 0, 1, 2.4 },   { 0, 4, 4.6 },   { 1, 3, 3.1 },   { 2, 1, 1.2 },  { 2, 8, 5.7 },   { 3, 5, 3.8 },
        { 3, 6, 2.9 },   { 4, 6, 5.5 },   { 4, 11, 8.2 },  { 5, 9, 4.4 },  { 6, 5, 1.6 },   { 6, 10, 7.3 },
        { 7, 2, 1.9 },   { 7, 13, 6.1 },  { 8, 7, 3.3 },   { 8, 9, 2.7 },  { 9, 12, 4.8 },  { 10, 9, 2.5 },
        { 10, 14, 6.6 }, { 11, 12, 3.7 }, { 12, 10, 3.9 }, { 13, 8, 4.0 }, { 13, 12, 5.1 }, { 14, 13, 2.8 } } );

   VectorType distances( graph.getVertexCount() );
   std::vector< VectorType > expectedDistances = {
      { 0.0, 2.4, 34.3, 5.5, 4.6, 9.3, 8.4, 32.4, 29.1, 13.7, 15.7, 12.8, 16.5, 25.1, 22.3 },
      { -1.0, 0.0, 31.9, 3.1, -1.0, 6.9, 6.0, 30.0, 26.7, 11.3, 13.3, -1.0, 16.1, 22.7, 19.9 },
      { -1.0, 1.2, 0.0, 4.3, -1.0, 8.1, 7.2, 9.0, 5.7, 8.4, 14.5, -1.0, 13.2, 15.1, 21.1 },
      { -1.0, 30.0, 28.8, 0.0, -1.0, 3.8, 2.9, 26.9, 23.6, 8.2, 10.2, -1.0, 13.0, 19.6, 16.8 },
      { -1.0, 32.6, 31.4, 35.7, 0.0, 7.1, 5.5, 29.5, 26.2, 11.5, 12.8, 8.2, 11.9, 22.2, 19.4 },
      { -1.0, 32.9, 31.7, 36.0, -1.0, 0.0, 38.9, 29.8, 26.5, 4.4, 13.1, -1.0, 9.2, 22.5, 19.7 },
      { -1.0, 27.1, 25.9, 30.2, -1.0, 1.6, 0.0, 24.0, 20.7, 6.0, 7.3, -1.0, 10.8, 16.7, 13.9 },
      { -1.0, 3.1, 1.9, 6.2, -1.0, 10.0, 9.1, 0.0, 7.6, 10.3, 15.1, -1.0, 11.2, 6.1, 21.7 },
      { -1.0, 6.4, 5.2, 9.5, -1.0, 13.3, 12.4, 3.3, 0.0, 2.7, 11.4, -1.0, 7.5, 9.4, 18.0 },
      { -1.0, 28.5, 27.3, 31.6, -1.0, 35.4, 34.5, 25.4, 22.1, 0.0, 8.7, -1.0, 4.8, 18.1, 15.3 },
      { -1.0, 19.8, 18.6, 22.9, -1.0, 26.7, 25.8, 16.7, 13.4, 2.5, 0.0, -1.0, 7.3, 9.4, 6.6 },
      { -1.0, 27.4, 26.2, 30.5, -1.0, 34.3, 33.4, 24.3, 21.0, 10.1, 7.6, 0.0, 3.7, 17.0, 14.2 },
      { -1.0, 23.7, 22.5, 26.8, -1.0, 30.6, 29.7, 20.6, 17.3, 6.4, 3.9, -1.0, 0.0, 13.3, 10.5 },
      { -1.0, 10.4, 9.2, 13.5, -1.0, 17.3, 16.4, 7.3, 4.0, 6.7, 9.0, -1.0, 5.1, 0.0, 15.6 },
      { -1.0, 13.2, 12.0, 16.3, -1.0, 20.1, 19.2, 10.1, 6.8, 9.5, 11.8, -1.0, 7.9, 2.8, 0.0 }
   };

   for( int start_node = 0; start_node < graph.getVertexCount(); start_node++ ) {
      TNL::Graphs::Algorithms::singleSourceShortestPath( graph, start_node, distances );
      for( IndexType i = 0; i < graph.getVertexCount(); i++ )
         ASSERT_FLOAT_EQ( distances.getElement( i ), expectedDistances[ start_node ].getElement( i ) )
            << "start_node: " << start_node << " distances[ " << i << " ]: " << distances.getElement( i )
            << " expectedDistances[ " << start_node << " ][ " << i << " ]: " << expectedDistances[ start_node ].getElement( i )
            << " distances: " << distances << " expectedDistances[ " << start_node << " ]: " << expectedDistances[ start_node ];
   }
}

TYPED_TEST( GraphTest, test_SSSP_withIndexes_inducedSubgraph )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 }, { 0, 4, 0.5 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 4, 3, 0.5 },
      } );
   // clang-format on
   const TNL::Containers::Vector< IndexType, DeviceType, IndexType > vertexIndexes( { 0, 1, 3 } );
   const VectorType expectedDistances( { 0.0, 1.0, -1.0, -1.0, -1.0 } );
   VectorType distances;

   TNL::Graphs::Algorithms::singleSourceShortestPath( graph, 0, vertexIndexes, distances );

   for( IndexType i = 0; i < graph.getVertexCount(); i++ )
      ASSERT_FLOAT_EQ( distances.getElement( i ), expectedDistances.getElement( i ) );
}

TYPED_TEST( GraphTest, test_SSSPIf_inducedSubgraph )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 }, { 0, 4, 0.5 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 4, 3, 0.5 },
      } );
   // clang-format on
   const VectorType expectedDistances( { 0.0, 1.0, 2.0, -1.0, -1.0 } );
   VectorType distances;
   const auto firstThreeVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex <= 2;
   };

   TNL::Graphs::Algorithms::singleSourceShortestPathIf( graph, 0, firstThreeVertices, distances );

   for( IndexType i = 0; i < graph.getVertexCount(); i++ )
      ASSERT_FLOAT_EQ( distances.getElement( i ), expectedDistances.getElement( i ) );
}

TYPED_TEST( GraphTest, test_SSSP_byEdges_wholeGraph )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      4,
      {
         { 0, 1, 1.0 }, { 0, 2, 5.0 },
         { 1, 2, 1.0 }, { 1, 3, 10.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   const VectorType expectedDistances( { 0.0, 1.0, 3.0, 4.0 } );
   VectorType distances;
   const auto transformWeights = [ = ] __cuda_callable__( IndexType source, IndexType target, RealType weight )
   {
      if( source == 0 && target == 2 )
         return std::numeric_limits< RealType >::infinity();
      if( source == 1 && target == 2 )
         return weight * static_cast< RealType >( 2.0 );
      return weight;
   };

   TNL::Graphs::Algorithms::singleSourceShortestPath( graph, 0, transformWeights, distances );

   for( IndexType i = 0; i < graph.getVertexCount(); i++ )
      ASSERT_FLOAT_EQ( distances.getElement( i ), expectedDistances.getElement( i ) );
}

TYPED_TEST( GraphTest, test_SSSP_byEdgesIf_inducedSubgraph )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5,
      {
         { 0, 1, 1.0 }, { 0, 2, 5.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 3, 4, 1.0 },
      } );
   // clang-format on

   const VectorType expectedDistances( { 0.0, 1.0, -1.0, -1.0, -1.0 } );
   VectorType distances;
   const auto firstFourVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex <= 3;
   };
   const auto blockEdgesToTwo = [ = ] __cuda_callable__( IndexType source, IndexType target, RealType weight )
   {
      if( source == 0 && target == 2 )
         return std::numeric_limits< RealType >::infinity();
      if( source == 1 && target == 2 )
         return std::numeric_limits< RealType >::infinity();
      return weight;
   };

   TNL::Graphs::Algorithms::singleSourceShortestPathIf(
      graph, static_cast< IndexType >( 0 ), firstFourVertices, blockEdgesToTwo, distances );

   for( IndexType i = 0; i < graph.getVertexCount(); i++ )
      ASSERT_FLOAT_EQ( distances.getElement( i ), expectedDistances.getElement( i ) );
}

TYPED_TEST( GraphTest, test_SSSP_withInactiveStart_throws )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   const GraphType graph( 4, { { 0, 1, 1.0 }, { 1, 2, 1.0 }, { 2, 3, 1.0 } } );
   const TNL::Containers::Vector< IndexType, DeviceType, IndexType > vertexIndexes( { 0, 1, 2 } );
   VectorType distances;

   EXPECT_THROW(
      TNL::Graphs::Algorithms::singleSourceShortestPath( graph, static_cast< IndexType >( 3 ), vertexIndexes, distances ),
      std::invalid_argument );
}

// clang-format off
// Weighted directed graph A (10 vertices, symmetric adjacency, varied weights).
// Used as the common "large" graph for subgraph cross-validation tests.
//
//     0---1---2
//     |   |   |
//     3---4---5
//     |   |   |
//     6---7---8---9
//
// Edge weights (both directions):
//   0-1:0.5, 0-3:1.0, 1-2:0.5, 1-4:2.0, 2-5:0.5,
//   3-4:0.8, 3-6:0.5, 4-5:2.0, 4-7:0.8, 5-8:0.5,
//   6-7:0.5, 7-8:0.5, 8-9:0.5
// clang-format on

template< typename GraphType >
GraphType
makeWeightedDirectedGraphA()
{
   using Real = typename GraphType::ValueType;
   // clang-format off
   return GraphType(
      10,
      {
         { 0, 1, Real( 0.5 ) }, { 0, 3, Real( 1.0 ) },
         { 1, 0, Real( 0.5 ) }, { 1, 2, Real( 0.5 ) }, { 1, 4, Real( 2.0 ) },
         { 2, 1, Real( 0.5 ) }, { 2, 5, Real( 0.5 ) },
         { 3, 0, Real( 1.0 ) }, { 3, 4, Real( 0.8 ) }, { 3, 6, Real( 0.5 ) },
         { 4, 1, Real( 2.0 ) }, { 4, 3, Real( 0.8 ) }, { 4, 5, Real( 2.0 ) }, { 4, 7, Real( 0.8 ) },
         { 5, 2, Real( 0.5 ) }, { 5, 4, Real( 2.0 ) }, { 5, 8, Real( 0.5 ) },
         { 6, 3, Real( 0.5 ) }, { 6, 7, Real( 0.5 ) },
         { 7, 4, Real( 0.8 ) }, { 7, 6, Real( 0.5 ) }, { 7, 8, Real( 0.5 ) },
         { 8, 5, Real( 0.5 ) }, { 8, 7, Real( 0.5 ) }, { 8, 9, Real( 0.5 ) },
         { 9, 8, Real( 0.5 ) },
      } );
   // clang-format on
}

template< typename GraphType >
GraphType
makeWeightedSubgraphB()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,3,4,6,7,9} -> remapped to {0,1,2,3,4,5,6}
   // clang-format off
   return GraphType(
      7,
      {
         { 0, 1, Real( 0.5 ) }, { 0, 2, Real( 1.0 ) },
         { 1, 0, Real( 0.5 ) }, { 1, 3, Real( 2.0 ) },
         { 2, 0, Real( 1.0 ) }, { 2, 3, Real( 0.8 ) }, { 2, 4, Real( 0.5 ) },
         { 3, 1, Real( 2.0 ) }, { 3, 2, Real( 0.8 ) }, { 3, 5, Real( 0.8 ) },
         { 4, 2, Real( 0.5 ) }, { 4, 5, Real( 0.5 ) },
         { 5, 3, Real( 0.8 ) }, { 5, 4, Real( 0.5 ) },
      } );
   // clang-format on
}

template< typename GraphType >
GraphType
makeWeightedSubgraphD()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,2,3,5,6,7,8,9} -> remapped to {0,1,2,3,4,5,6,7,8}
   // clang-format off
   return GraphType(
      9,
      {
         { 0, 1, Real( 0.5 ) }, { 0, 3, Real( 1.0 ) },
         { 1, 0, Real( 0.5 ) }, { 1, 2, Real( 0.5 ) },
         { 2, 1, Real( 0.5 ) }, { 2, 4, Real( 0.5 ) },
         { 3, 0, Real( 1.0 ) }, { 3, 5, Real( 0.5 ) },
         { 4, 2, Real( 0.5 ) }, { 4, 7, Real( 0.5 ) },
         { 5, 3, Real( 0.5 ) }, { 5, 6, Real( 0.5 ) },
         { 6, 5, Real( 0.5 ) }, { 6, 7, Real( 0.5 ) },
         { 7, 4, Real( 0.5 ) }, { 7, 6, Real( 0.5 ) }, { 7, 8, Real( 0.5 ) },
         { 8, 7, Real( 0.5 ) },
      } );
   // clang-format on
}

template< typename GraphType >
GraphType
makeWeightedSubgraphC()
{
   using Real = typename GraphType::ValueType;
   // All 10 vertices, edges {0,3} and {3,0} removed.
   // clang-format off
   return GraphType(
      10,
      {
         { 0, 1, Real( 0.5 ) },
         { 1, 0, Real( 0.5 ) }, { 1, 2, Real( 0.5 ) }, { 1, 4, Real( 2.0 ) },
         { 2, 1, Real( 0.5 ) }, { 2, 5, Real( 0.5 ) },
         { 3, 4, Real( 0.8 ) }, { 3, 6, Real( 0.5 ) },
         { 4, 1, Real( 2.0 ) }, { 4, 3, Real( 0.8 ) }, { 4, 5, Real( 2.0 ) }, { 4, 7, Real( 0.8 ) },
         { 5, 2, Real( 0.5 ) }, { 5, 4, Real( 2.0 ) }, { 5, 8, Real( 0.5 ) },
         { 6, 3, Real( 0.5 ) }, { 6, 7, Real( 0.5 ) },
         { 7, 4, Real( 0.8 ) }, { 7, 6, Real( 0.5 ) }, { 7, 8, Real( 0.5 ) },
         { 8, 5, Real( 0.5 ) }, { 8, 7, Real( 0.5 ) }, { 8, 9, Real( 0.5 ) },
         { 9, 8, Real( 0.5 ) },
      } );
   // clang-format on
}

template< typename GraphType >
GraphType
makeWeightedSubgraphE2()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,3,4,6,7} -> remapped to {0,1,2,3,4,5}
   // Edges {0,3} and {3,0} also removed.
   // clang-format off
   return GraphType(
      6,
      {
         { 0, 1, Real( 0.5 ) },
         { 1, 0, Real( 0.5 ) }, { 1, 3, Real( 2.0 ) },
         { 2, 3, Real( 0.8 ) }, { 2, 4, Real( 0.5 ) },
         { 3, 1, Real( 2.0 ) }, { 3, 2, Real( 0.8 ) }, { 3, 5, Real( 0.8 ) },
         { 4, 2, Real( 0.5 ) }, { 4, 5, Real( 0.5 ) },
         { 5, 3, Real( 0.8 ) }, { 5, 4, Real( 0.5 ) },
      } );
   // clang-format on
}

template< typename VectorType >
void
remapAndCompareFloatDistances( const VectorType& distA, const VectorType& distB, const std::vector< int >& newToOld )
{
   using RealType = typename VectorType::ValueType;
   for( int i = 0; i < (int) newToOld.size(); i++ ) {
      RealType expected = distB.getElement( i );
      RealType actual = distA.getElement( newToOld[ i ] );
      if( expected < 0 && actual < 0 )
         continue;
      ASSERT_FLOAT_EQ( actual, expected ) << "vertex " << newToOld[ i ] << " (subgraph idx " << i << ")";
   }
}

TYPED_TEST( GraphTest, test_SSSP_subgraph_vertex_removal_predicate )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeWeightedDirectedGraphA< GraphType >();
   const auto subgraphB = makeWeightedSubgraphB< GraphType >();

   const auto excludeVertices = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 2 && v != 5 && v != 8;
   };

   VectorType distA, distB;
   TNL::Graphs::Algorithms::singleSourceShortestPathIf( graphA, 0, excludeVertices, distA );
   TNL::Graphs::Algorithms::singleSourceShortestPath( subgraphB, 0, distB );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   remapAndCompareFloatDistances( distA, distB, newToOld );

   EXPECT_FLOAT_EQ( distA.getElement( 2 ), RealType( -1 ) );
   EXPECT_FLOAT_EQ( distA.getElement( 5 ), RealType( -1 ) );
   EXPECT_FLOAT_EQ( distA.getElement( 8 ), RealType( -1 ) );
}

TYPED_TEST( GraphTest, test_SSSP_subgraph_vertex_removal_indexed )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeWeightedDirectedGraphA< GraphType >();
   const auto subgraphB = makeWeightedSubgraphB< GraphType >();

   const TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType > vertexIndexes(
      { 0, 1, 3, 4, 6, 7, 9 } );

   VectorType distA, distB;
   TNL::Graphs::Algorithms::singleSourceShortestPath( graphA, 0, vertexIndexes, distA );
   TNL::Graphs::Algorithms::singleSourceShortestPath( subgraphB, 0, distB );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   remapAndCompareFloatDistances( distA, distB, newToOld );

   EXPECT_FLOAT_EQ( distA.getElement( 2 ), RealType( -1 ) );
   EXPECT_FLOAT_EQ( distA.getElement( 5 ), RealType( -1 ) );
   EXPECT_FLOAT_EQ( distA.getElement( 8 ), RealType( -1 ) );
}

TYPED_TEST( GraphTest, test_SSSP_subgraph_vertex_removal_disconnected )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeWeightedDirectedGraphA< GraphType >();
   const auto subgraphD = makeWeightedSubgraphD< GraphType >();

   const auto excludeFour = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 4;
   };

   VectorType distA, distD;
   TNL::Graphs::Algorithms::singleSourceShortestPathIf( graphA, 0, excludeFour, distA );
   TNL::Graphs::Algorithms::singleSourceShortestPath( subgraphD, 0, distD );

   const std::vector< int > newToOld = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
   remapAndCompareFloatDistances( distA, distD, newToOld );

   EXPECT_FLOAT_EQ( distA.getElement( 4 ), RealType( -1 ) );
}

TYPED_TEST( GraphTest, test_SSSP_subgraph_edge_removal_wholeGraph )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeWeightedDirectedGraphA< GraphType >();
   const auto subgraphC = makeWeightedSubgraphC< GraphType >();

   const auto blockEdge03 = [ = ] __cuda_callable__( IndexType source, IndexType target, RealType weight )
   {
      if( ( source == 0 && target == 3 ) || ( source == 3 && target == 0 ) )
         return std::numeric_limits< RealType >::infinity();
      return weight;
   };

   VectorType distA, distC;
   TNL::Graphs::Algorithms::singleSourceShortestPath( graphA, 0, blockEdge03, distA );
   TNL::Graphs::Algorithms::singleSourceShortestPath( subgraphC, 0, distC );

   for( IndexType i = 0; i < graphA.getVertexCount(); i++ ) {
      RealType a = distA.getElement( i );
      RealType c = distC.getElement( i );
      if( a < 0 && c < 0 )
         continue;
      ASSERT_FLOAT_EQ( a, c ) << "vertex " << i;
   }
}

TYPED_TEST( GraphTest, test_SSSP_subgraph_edge_removal_withIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeWeightedDirectedGraphA< GraphType >();
   const auto subgraphE2 = makeWeightedSubgraphE2< GraphType >();

   const TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType > vertexIndexes( { 0, 1, 3, 4, 6, 7 } );

   const auto blockEdge03 = [ = ] __cuda_callable__( IndexType source, IndexType target, RealType weight )
   {
      if( ( source == 0 && target == 3 ) || ( source == 3 && target == 0 ) )
         return std::numeric_limits< RealType >::infinity();
      return weight;
   };

   VectorType distA, distE2;
   TNL::Graphs::Algorithms::singleSourceShortestPath( graphA, 0, vertexIndexes, blockEdge03, distA );
   TNL::Graphs::Algorithms::singleSourceShortestPath( subgraphE2, 0, distE2 );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7 };
   remapAndCompareFloatDistances( distA, distE2, newToOld );

   EXPECT_FLOAT_EQ( distA.getElement( 2 ), RealType( -1 ) );
   EXPECT_FLOAT_EQ( distA.getElement( 5 ), RealType( -1 ) );
   EXPECT_FLOAT_EQ( distA.getElement( 8 ), RealType( -1 ) );
   EXPECT_FLOAT_EQ( distA.getElement( 9 ), RealType( -1 ) );
}

// Regression test for the sequential Dijkstra lazy-deletion invariant.
//
// The re-emplace of a vertex already in the priority queue and the stale-entry
// guard (`if( currentDistance > distances[ current ] ) continue;`) live inside
// the `if constexpr( DeviceType == Devices::Sequential )` branch of
// singleSourceShortestPath_impl, so only the Sequential type exercises them
// directly.  The Host type dispatches to parallelSingleSourceShortestPath
// (atomicMin / fetch_min, no priority queue); this TYPED_TEST validates its
// parallel-relaxation correctness on the same graph.  Both runs are valuable.
//
// Pinned graph (vertex 4 is isolated, to also lock the -1 unreachable
// sentinel):  0->1 (w=1), 0->2 (w=5), 1->3 (w=10), 2->3 (w=1)
// Expected relaxation/pop order from source 0 (sequential backend):
//   pop (0,0)  -> relax 0->1: dist[1] = 1,  emplace (1,1)
//              -> relax 0->2: dist[2] = 5,  emplace (5,2)
//   pop (1,1)  -> relax 1->3: dist[3] = 11, emplace (11,3)   [first discovery]
//   pop (5,2)  -> relax 2->3: dist[3] = 6,  emplace (6,3)    [re-emplace, smaller]
//   pop (6,3)  -> currentDistance == distances[3], processed (no out-edges)
//   pop (11,3) -> currentDistance > distances[3]  => skip    [lazy-deletion guard]
// Hand-computed expected distances: {0, 1, 5, 6, -1}.
TYPED_TEST( GraphTest, test_SSSP_multi_relaxation )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph(
      5, // vertex 4 is isolated -> unreachable sentinel (-1)
      {
         { 0, 1, 1.0 }, { 0, 2, 5.0 },
         { 1, 3, 10.0 },
         { 2, 3, 1.0 },
      } );
   // clang-format on

   const VectorType expectedDistances( { 0.0, 1.0, 5.0, 6.0, -1.0 } );
   VectorType distances( graph.getVertexCount() );

   TNL::Graphs::Algorithms::singleSourceShortestPath( graph, 0, distances );

   for( IndexType i = 0; i < graph.getVertexCount(); i++ )
      ASSERT_FLOAT_EQ( distances.getElement( i ), expectedDistances.getElement( i ) ) << "vertex " << i;
}

#include "../../main.h"
