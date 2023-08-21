#include <iostream>
#include <cstdint>

#include <TNL/Graphs/minimumSpanningTree.h>
#include <TNL/Graphs/trees.h>
#include <TNL/Graphs/GraphWriter.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Containers/StaticVector.h>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class GraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using GraphType = TNL::Graphs::Graph< MatrixType, TNL::Graphs::GraphTypes::Undirected >;
};

// types for which MatrixTest is instantiated
using GraphTestTypes = ::testing::Types< TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int >,
                                         TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >
#ifdef __CUDACC__
//,TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >
#endif
                                         >;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

TYPED_TEST( GraphTest, test_MST_small )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;

   // Create a sample graph.
   // clang-format off
   GraphType graph( 6, // number of graph nodes
        {  // matrix elements definition
                        { 0, 1, 1.0 },  { 0, 2, 3.0 },
         { 1, 0, 1.0 },                 { 1, 2, 2.0 },
         { 2, 0, 3.0 }, { 2, 1, 2.0 },
                                                                    { 3, 4, 4.0 }, { 3, 5, 5.0 },
                                                      { 4, 3, 4.0 },               { 4, 5, 3.0 },
                                                      { 5, 3, 5.0 }, { 5, 4, 3.0 }
        });
   // clang-format on

   GraphType expected_tree( 6, { { 1, 0, 1.0 }, { 2, 1, 2.0 }, { 4, 3, 4.0 }, { 5, 4, 3.0 } } );

   GraphType minimum_tree;
   TNL::Containers::Vector< IndexType > roots;
   TNL::Graphs::minimumSpanningTree( graph, minimum_tree, roots );
   minimum_tree.getAdjacencyMatrix().sortColumnIndexes();
   ASSERT_EQ( minimum_tree, expected_tree );
}

TYPED_TEST( GraphTest, test_MST_medium )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;

   // Create a sample graph.
   // clang-format off
   GraphType graph( 9, // number of graph nodes
        {
           { 1, 0, 4.0 },
                          { 2, 1, 8.0 },
                                          { 3, 2, 7.0 },
                                                         { 4, 3,  9.0 },
                                          { 5, 2, 4.0 }, { 5, 3, 14.0 }, { 5, 4, 10.0 },
                                                                                         { 6, 5,  2.0 },
           { 7, 0, 8.0 }, { 7, 1, 11.0 },                                                               { 7, 6, 1.0 },
                                          { 8, 2, 2.0 },                                                { 8, 6, 6.0 },  { 8, 7,  7.0 }
         } );
   // clang-format on

   GraphType expectedTree( 9,
                           { { 1, 0, 4.0 },
                             { 2, 1, 8.0 },
                             { 3, 2, 7.0 },
                             { 4, 3, 9.0 },
                             { 5, 2, 4.0 },
                             { 6, 5, 2.0 },
                             { 7, 6, 1.0 },
                             { 8, 2, 2.0 } } );

   GraphType minimum_tree;
   TNL::Containers::Vector< IndexType > roots;
   TNL::Graphs::minimumSpanningTree( graph, minimum_tree, roots );
   minimum_tree.getAdjacencyMatrix().sortColumnIndexes();
   //const auto& v1 = minimum_tree.getAdjacencyMatrix().getValues();
   //const auto& v2 = expectedTree.getAdjacencyMatrix().getValues();
   ASSERT_EQ( minimum_tree, expectedTree );
}

TYPED_TEST( GraphTest, test_MST_large )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;

   // Create a sample graph.
   std::map< std::pair< IndexType, IndexType >, RealType > map{
      { { 1, 0 }, 3.0 },   { { 5, 0 }, 12.0 },   { { 2, 1 }, 7.0 },    { { 6, 1 }, 11.0 },   { { 3, 2 }, 4.0 },
      { { 7, 2 }, 8.0 },   { { 6, 2 }, 2.0 },    { { 4, 3 }, 9.0 },    { { 7, 3 }, 3.0 },    { { 8, 3 }, 6.0 },
      { { 8, 4 }, 1.0 },   { { 9, 4 }, 14.0 },   { { 6, 5 }, 5.0 },    { { 10, 5 }, 15.0 },  { { 7, 6 }, 10.0 },
      { { 11, 6 }, 6.0 },  { { 12, 7 }, 13.0 },  { { 9, 8 }, 6.0 },    { { 13, 8 }, 7.0 },   { { 14, 9 }, 3.0 },
      { { 11, 10 }, 9.0 }, { { 15, 10 }, 8.0 },  { { 12, 11 }, 5.0 },  { { 16, 11 }, 7.0 },  { { 13, 12 }, 8.0 },
      { { 17, 12 }, 4.0 }, { { 18, 13 }, 12.0 }, { { 19, 14 }, 10.0 }, { { 16, 15 }, 11.0 }, { { 17, 16 }, 2.0 },
      { { 18, 17 }, 3.0 }, { { 19, 18 }, 9.0 }
   };
   GraphType graph( 20, map );

   GraphType expected_tree( 20,
                            { { 8, 4, 1.0 },
                              { 6, 2, 2.0 },
                              { 17, 16, 2.0 },
                              { 1, 0, 3.0 },
                              { 7, 3, 3.0 },
                              { 14, 9, 3.0 },
                              { 18, 17, 3.0 },
                              { 3, 2, 4.0 },
                              { 17, 12, 4.0 },
                              { 6, 5, 5.0 },
                              { 12, 11, 5.0 },
                              { 11, 6, 6.0 },
                              { 8, 3, 6.0 },
                              { 9, 8, 6.0 },
                              { 2, 1, 7.0 },
                              { 13, 8, 7.0 },
                              { 15, 10, 8.0 },
                              { 11, 10, 9.0 },
                              { 19, 18, 9.0 } } );

   GraphType minimum_tree;
   TNL::Containers::Vector< IndexType > roots;
   TNL::Graphs::minimumSpanningTree( graph, minimum_tree, roots );
   minimum_tree.getAdjacencyMatrix().sortColumnIndexes();
   //const auto& v1 = minimum_tree.getAdjacencyMatrix().getValues();
   //const auto& v2 = expected_tree.getAdjacencyMatrix().getValues();
   ASSERT_EQ( minimum_tree, expected_tree );
}

TYPED_TEST( GraphTest, test_MST_large_2 )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;

   GraphType graph( 10,
                    { { 0, 1, 0.5 }, { 0, 2, 0.7 }, { 1, 3, 1.0 }, { 2, 4, 0.3 }, { 3, 4, 0.8 }, { 4, 5, 0.2 }, { 5, 6, 0.6 },
                      { 6, 7, 0.9 }, { 7, 8, 0.3 }, { 8, 9, 0.5 }, { 9, 0, 0.7 }, { 1, 5, 0.5 }, { 2, 6, 0.6 }, { 3, 7, 0.3 },
                      { 4, 8, 0.4 }, { 5, 9, 0.9 }, { 6, 2, 0.8 }, { 7, 3, 0.2 }, { 8, 1, 0.7 }, { 9, 4, 0.1 } } );

   TNL::Graphs::GraphWriter< GraphType >::writeEdgeList( "graph-10-30.lst", graph );
   GraphType minimum_tree;
   TNL::Containers::Vector< IndexType > roots;
   TNL::Graphs::minimumSpanningTree( graph, minimum_tree, roots );
   TNL::Graphs::GraphWriter< GraphType >::writeEdgeList( "graph-10-30-mst.lst", minimum_tree );
   ASSERT_TRUE( TNL::Graphs::isTree( minimum_tree ) );
   ASSERT_NEAR( minimum_tree.getTotalWeight(), 3.1, 0.0001 );
}

TYPED_TEST( GraphTest, test_MST_large_3 )
{
   using GraphType = typename TestFixture::GraphType;
   using RealType = typename GraphType::ValueType;
   using IndexType = typename GraphType::IndexType;

   // Create a TNL::Matrices::SparseMatrix for a sample graph.
   std::map< std::pair< IndexType, IndexType >, RealType > map{
      { { 3, 0 }, 10.0 },   { { 4, 2 }, 7.0 },    { { 5, 0 }, 5.0 },    { { 5, 1 }, 1.0 },    { { 5, 3 }, 3.0 },
      { { 6, 1 }, 6.0 },    { { 6, 2 }, 4.0 },    { { 6, 0 }, 4.0 },    { { 7, 0 }, 4.0 },    { { 7, 2 }, 5.0 },
      { { 7, 3 }, 10.0 },   { { 7, 4 }, 9.0 },    { { 8, 5 }, 10.0 },   { { 9, 7 }, 8.0 },    { { 10, 5 }, 10.0 },
      { { 10, 4 }, 9.0 },   { { 10, 7 }, 5.0 },   { { 10, 3 }, 2.0 },   { { 10, 2 }, 3.0 },   { { 10, 0 }, 8.0 },
      { { 10, 1 }, 10.0 },  { { 10, 6 }, 7.0 },   { { 11, 3 }, 4.0 },   { { 11, 5 }, 8.0 },   { { 11, 0 }, 1.0 },
      { { 11, 6 }, 3.0 },   { { 11, 8 }, 4.0 },   { { 11, 2 }, 8.0 },   { { 11, 1 }, 9.0 },   { { 11, 4 }, 9.0 },
      { { 12, 7 }, 9.0 },   { { 12, 1 }, 5.0 },   { { 12, 2 }, 3.0 },   { { 12, 8 }, 1.0 },   { { 12, 6 }, 5.0 },
      { { 12, 10 }, 2.0 },  { { 12, 3 }, 9.0 },   { { 12, 4 }, 2.0 },   { { 12, 5 }, 7.0 },   { { 14, 5 }, 6.0 },
      { { 14, 10 }, 3.0 },  { { 14, 4 }, 9.0 },   { { 14, 9 }, 8.0 },   { { 14, 3 }, 3.0 },   { { 14, 8 }, 3.0 },
      { { 16, 5 }, 10.0 },  { { 16, 12 }, 5.0 },  { { 17, 11 }, 7.0 },  { { 17, 5 }, 9.0 },   { { 17, 13 }, 4.0 },
      { { 17, 9 }, 3.0 },   { { 17, 15 }, 4.0 },  { { 18, 16 }, 10.0 }, { { 18, 7 }, 7.0 },   { { 18, 13 }, 3.0 },
      { { 18, 8 }, 9.0 },   { { 18, 6 }, 4.0 },   { { 18, 1 }, 9.0 },   { { 18, 12 }, 2.0 },  { { 18, 14 }, 2.0 },
      { { 18, 11 }, 5.0 },  { { 18, 3 }, 1.0 },   { { 18, 2 }, 6.0 },   { { 19, 10 }, 10.0 }, { { 19, 4 }, 1.0 },
      { { 19, 8 }, 9.0 },   { { 19, 2 }, 3.0 },   { { 19, 7 }, 2.0 },   { { 19, 13 }, 1.0 },  { { 19, 16 }, 8.0 },
      { { 19, 14 }, 10.0 }, { { 19, 1 }, 3.0 },   { { 20, 12 }, 8.0 },  { { 20, 17 }, 5.0 },  { { 20, 0 }, 9.0 },
      { { 20, 3 }, 3.0 },   { { 22, 20 }, 9.0 },  { { 22, 2 }, 4.0 },   { { 22, 0 }, 3.0 },   { { 22, 9 }, 6.0 },
      { { 22, 18 }, 9.0 },  { { 23, 21 }, 3.0 },  { { 23, 19 }, 8.0 },  { { 23, 20 }, 9.0 },  { { 23, 1 }, 6.0 },
      { { 23, 17 }, 1.0 },  { { 23, 4 }, 6.0 },   { { 23, 10 }, 2.0 },  { { 23, 16 }, 1.0 },  { { 23, 7 }, 10.0 },
      { { 23, 18 }, 3.0 },  { { 23, 0 }, 5.0 },   { { 23, 14 }, 2.0 },  { { 23, 12 }, 5.0 },  { { 23, 6 }, 3.0 },
      { { 23, 9 }, 9.0 },   { { 23, 15 }, 10.0 }, { { 24, 16 }, 4.0 },  { { 24, 21 }, 8.0 },  { { 24, 5 }, 3.0 },
      { { 24, 19 }, 8.0 },  { { 24, 20 }, 7.0 },  { { 24, 15 }, 6.0 },  { { 24, 3 }, 10.0 },  { { 24, 2 }, 10.0 },
      { { 24, 7 }, 4.0 },   { { 25, 3 }, 8.0 },   { { 25, 18 }, 1.0 },  { { 25, 4 }, 8.0 },   { { 25, 13 }, 5.0 },
      { { 25, 1 }, 2.0 },   { { 25, 17 }, 10.0 }, { { 25, 15 }, 1.0 },  { { 25, 6 }, 3.0 },   { { 25, 5 }, 5.0 },
      { { 25, 7 }, 9.0 },   { { 25, 11 }, 7.0 },  { { 25, 21 }, 10.0 }, { { 25, 10 }, 5.0 },  { { 25, 22 }, 4.0 },
      { { 25, 19 }, 7.0 },  { { 25, 8 }, 8.0 },   { { 25, 12 }, 4.0 },  { { 25, 23 }, 5.0 },  { { 25, 20 }, 1.0 },
      { { 25, 14 }, 10.0 }, { { 25, 2 }, 6.0 },   { { 25, 0 }, 4.0 },   { { 26, 8 }, 1.0 },   { { 26, 12 }, 5.0 },
      { { 26, 21 }, 4.0 },  { { 27, 12 }, 8.0 },  { { 27, 8 }, 4.0 },   { { 28, 3 }, 6.0 },   { { 28, 14 }, 4.0 },
      { { 28, 22 }, 7.0 },  { { 28, 5 }, 3.0 },   { { 28, 8 }, 10.0 },  { { 28, 1 }, 6.0 },   { { 28, 18 }, 7.0 },
      { { 28, 24 }, 2.0 },  { { 28, 19 }, 9.0 },  { { 28, 17 }, 2.0 },  { { 28, 2 }, 1.0 },   { { 28, 23 }, 9.0 },
      { { 28, 11 }, 1.0 },  { { 28, 15 }, 10.0 }, { { 28, 7 }, 3.0 }
   };

   GraphType graph( 30, map );

   GraphType expected_tree( 30, { { 18, 3, 1.0 },  { 11, 0, 1.0 },  { 19, 4, 1.0 },  { 28, 2, 1.0 },  { 1, 5, 1.0 },
                                  { 12, 8, 1.0 },  { 26, 8, 1.0 },  { 28, 11, 1.0 }, { 23, 16, 1.0 }, { 23, 17, 1.0 },
                                  { 19, 13, 1.0 }, { 25, 15, 1.0 }, { 25, 18, 1.0 }, { 25, 20, 1.0 }, { 10, 3, 2.0 },
                                  { 12, 4, 2.0 },  { 25, 1, 2.0 },  { 19, 7, 2.0 },  { 12, 10, 2.0 }, { 23, 10, 2.0 },
                                  { 18, 14, 2.0 }, { 28, 17, 2.0 }, { 28, 24, 2.0 }, { 22, 0, 3.0 },  { 11, 6, 3.0 },
                                  { 17, 9, 3.0 },  { 21, 23, 3.0 }, { 27, 8, 4.0 } } );

   GraphType minimum_tree;
   TNL::Containers::Vector< IndexType > roots;
   TNL::Graphs::minimumSpanningTree( graph, minimum_tree, roots );
   minimum_tree.getAdjacencyMatrix().sortColumnIndexes();
   const auto& v1 = minimum_tree.getAdjacencyMatrix().getValues();
   const auto& v2 = expected_tree.getAdjacencyMatrix().getValues();
   ASSERT_TRUE( TNL::Graphs::isForest( minimum_tree ) );  // node 29 is not connected
   ASSERT_EQ( sum( maximum( v1, 0 ) ), sum( maximum( v2, 0 ) ) );
}

#include "../main.h"
