#include <TNL/Graphs/singleSourceShortestPath.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Containers/StaticVector.h>

#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class GraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::Graph< MatrixType, TNL::Graphs::GraphTypes::Directed >;
};

// types for which MatrixTest is instantiated
using GraphTestTypes = ::testing::Types< TNL::Matrices::SparseMatrix< float, TNL::Devices::Sequential, int >,
                                         TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int >
#ifdef __CUDACC__
                                         ,
                                         TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int >
#endif
                                         >;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

TYPED_TEST( GraphTest, test_BFS_small )
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

   VectorType distances( graph.getNodeCount() );
   std::vector< VectorType > expectedDistances{ { 0.0, 0.5, 1.2, 2.0, 3.5 },
                                                { 0.5, 0.0, 1.7, 2.3, 3.7 },
                                                { 1.2, 1.7, 0.0, 0.8, 2.3 },
                                                { 2.0, 2.3, 0.8, 0.0, 1.5 },
                                                { 3.5, 3.7, 2.3, 1.5, 0.0 } };

   for( int start_node = 0; start_node < graph.getNodeCount(); ++start_node ) {
      TNL::Graphs::singleSourceShortestPath( graph, start_node, distances );
      ASSERT_EQ( distances, expectedDistances[ start_node ] ) << "start_node: " << start_node;
   }
}

TYPED_TEST( GraphTest, test_BFS_larger )
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

   VectorType distances( graph.getNodeCount() );
   std::vector< VectorType > expectedDistances = {
      { 0.0, 0.5, 1.2, 2.0, 4.2, 3.3, 3.5, 5.1, 5.0, 6.8 },        { -1.0, 0.0, -1.0, 2.3, 3.7, -1.0, 3.8, 4.6, -1.0, 6.8 },
      { -1.0, -1.0, 0.0, 0.8, -1.0, 2.1, 2.3, -1.0, 3.8, 5.6 },    { -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, 1.5, -1.0, -1.0, 4.8 },
      { -1.0, 3.7, -1.0, 1.5, 0.0, -1.0, 3.0, 0.9, -1.0, 3.1 },    { -1.0, -1.0, -1.0, 4.3, -1.0, 0.0, 2.4, -1.0, 1.7, 3.6 },
      { -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 0.0, -1.0, -1.0, 3.3 }, { -1.0, 4.1, -1.0, 1.9, 0.4, -1.0, 3.4, 0.0, -1.0, 2.2 },
      { -1.0, -1.0, -1.0, 2.6, -1.0, -1.0, 4.1, -1.0, 0.0, 1.9 },  { -1.0, -1.0, -1.0, 0.7, -1.0, -1.0, 2.2, -1.0, -1.0, 0.0 }
   };

   for( int start_node = 0; start_node < graph.getNodeCount(); start_node++ ) {
      TNL::Graphs::singleSourceShortestPath( graph, start_node, distances );
      for( IndexType i = 0; i < graph.getNodeCount(); i++ )
         ASSERT_FLOAT_EQ( distances.getElement( i ), expectedDistances[ start_node ].getElement( i ) )
            << "start_node: " << start_node << " distances[ " << i << " ]: " << distances.getElement( i )
            << " expectedDistances[ " << start_node << " ][ " << i << " ]: " << expectedDistances[ start_node ].getElement( i )
            << " distances: " << distances << " expectedDistances[ " << start_node << " ]: " << expectedDistances[ start_node ];
   }
}

TYPED_TEST( GraphTest, test_BFS_largest )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   // Create a sample graph with 15 nodes.
   GraphType graph( 15,  // graph nodes count
                    {    // definition of graph edges
                      { 0, 1, 2.4 },   { 0, 4, 4.6 },   { 1, 3, 3.1 },   { 2, 1, 1.2 },  { 2, 8, 5.7 },   { 3, 5, 3.8 },
                      { 3, 6, 2.9 },   { 4, 6, 5.5 },   { 4, 11, 8.2 },  { 5, 9, 4.4 },  { 6, 5, 1.6 },   { 6, 10, 7.3 },
                      { 7, 2, 1.9 },   { 7, 13, 6.1 },  { 8, 7, 3.3 },   { 8, 9, 2.7 },  { 9, 12, 4.8 },  { 10, 9, 2.5 },
                      { 10, 14, 6.6 }, { 11, 12, 3.7 }, { 12, 10, 3.9 }, { 13, 8, 4.0 }, { 13, 12, 5.1 }, { 14, 13, 2.8 } } );

   VectorType distances( graph.getNodeCount() );
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

   for( int start_node = 0; start_node < graph.getNodeCount(); start_node++ ) {
      TNL::Graphs::singleSourceShortestPath( graph, start_node, distances );
      for( IndexType i = 0; i < graph.getNodeCount(); i++ )
         ASSERT_FLOAT_EQ( distances.getElement( i ), expectedDistances[ start_node ].getElement( i ) )
            << "start_node: " << start_node << " distances[ " << i << " ]: " << distances.getElement( i )
            << " expectedDistances[ " << start_node << " ][ " << i << " ]: " << expectedDistances[ start_node ].getElement( i )
            << " distances: " << distances << " expectedDistances[ " << start_node << " ]: " << expectedDistances[ start_node ];
   }
}

#include "../main.h"
