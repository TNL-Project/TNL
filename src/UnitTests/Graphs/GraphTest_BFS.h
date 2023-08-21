#include <TNL/Graphs/breadthFirstSearch.h>
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
   using GraphType = TNL::Graphs::Graph< MatrixType >;
};

// types for which MatrixTest is instantiated
using GraphTestTypes = ::testing::Types< TNL::Matrices::SparseMatrix< int, TNL::Devices::Sequential, int >,
                                         TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int >
#ifdef __CUDACC__
                                         ,
                                         TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int >
#endif
                                         >;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

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

   VectorType distances( graph.getNodeCount() );
   std::vector< VectorType > expectedDistances = {
      { 0, 1, 1, 2, 2 }, { 1, 0, 2, 1, 1 }, { 1, 2, 0, 1, 2 }, { 2, 1, 1, 0, 1 }, { 2, 1, 2, 1, 0 },
   };

   for( int start_node = 0; start_node < graph.getNodeCount(); ++start_node ) {
      TNL::Graphs::breadthFirstSearch( graph, start_node, distances );
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

   VectorType distances( graph.getNodeCount() );
   std::vector< VectorType > expectedDistances = {
      { 0, 1, 1, 2, 2, 2, 3, 3, 3, 4 }, { 1, 0, 2, 1, 1, 3, 2, 2, 3, 3 }, { 1, 2, 0, 1, 2, 1, 2, 3, 2, 3 },
      { 2, 1, 1, 0, 1, 2, 1, 2, 3, 2 }, { 2, 1, 2, 1, 0, 3, 2, 1, 2, 3 }, { 2, 3, 1, 2, 3, 0, 1, 2, 1, 2 },
      { 3, 2, 2, 1, 2, 1, 0, 3, 2, 1 }, { 3, 2, 3, 2, 1, 2, 3, 0, 1, 2 }, { 3, 3, 2, 3, 2, 1, 2, 1, 0, 1 },
      { 4, 3, 3, 2, 3, 2, 1, 2, 1, 0 },
   };

   for( int start_node = 0; start_node < graph.getNodeCount(); ++start_node ) {
      TNL::Graphs::breadthFirstSearch( graph, start_node, distances );
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
   GraphType graph( 15,  // number of graph nodes
                    {
                       // definition of edges
                       { 0, 1, 1.0 },   { 0, 3, 1.0 },   { 1, 0, 1.0 },   { 1, 2, 1.0 },   { 1, 4, 1.0 },   { 2, 1, 1.0 },
                       { 2, 5, 1.0 },   { 3, 0, 1.0 },   { 3, 4, 1.0 },   { 3, 6, 1.0 },   { 4, 1, 1.0 },   { 4, 3, 1.0 },
                       { 4, 5, 1.0 },   { 4, 7, 1.0 },   { 5, 2, 1.0 },   { 5, 4, 1.0 },   { 5, 8, 1.0 },   { 6, 3, 1.0 },
                       { 6, 7, 1.0 },   { 6, 9, 1.0 },   { 7, 4, 1.0 },   { 7, 6, 1.0 },   { 7, 8, 1.0 },   { 7, 10, 1.0 },
                       { 8, 5, 1.0 },   { 8, 7, 1.0 },   { 8, 11, 1.0 },  { 9, 6, 1.0 },   { 9, 10, 1.0 },  { 9, 12, 1.0 },
                       { 10, 7, 1.0 },  { 10, 9, 1.0 },  { 10, 11, 1.0 }, { 10, 13, 1.0 }, { 11, 8, 1.0 },  { 11, 10, 1.0 },
                       { 11, 14, 1.0 }, { 12, 9, 1.0 },  { 12, 13, 1.0 }, { 13, 10, 1.0 }, { 13, 12, 1.0 }, { 13, 14, 1.0 },
                       { 14, 11, 1.0 }, { 14, 13, 1.0 },
                    } );

   VectorType distances( graph.getNodeCount() );
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

   for( int start_node = 0; start_node < graph.getNodeCount(); start_node++ ) {
      TNL::Graphs::breadthFirstSearch( graph, start_node, distances );
      ASSERT_EQ( distances, expectedDistances[ start_node ] ) << "start_node: " << start_node;
   }
}

#include "../main.h"
