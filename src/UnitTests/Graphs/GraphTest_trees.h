#include <iostream>
#include <cstdint>
#include <TNL/Graphs/trees.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Graphs/Graph.h>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class GraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::Graph< MatrixType, TNL::Graphs::GraphTypes::Undirected >;
};

// types for which MatrixTest is instantiated
using GraphTestTypes =
   ::testing::Types< TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int >,
                     TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int, TNL::Matrices::SymmetricMatrix >,
                     TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >,
                     TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::SymmetricMatrix >
#ifdef __CUDACC__
                     ,
                     TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >,
                     TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::SymmetricMatrix >
#endif
                     >;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

TYPED_TEST( GraphTest, test_isTree_small )
{
   using GraphType = typename TestFixture::GraphType;
   // Create a sample graph.
   // clang-format off
   GraphType graph(
        10, // number of graph nodes
        {   // definition of graph edges
                       {0, 1, 1},    {0, 2, 1},
                                                {1, 3, 1},  {1, 4, 1},
                                                                        {2, 5, 1},  {2, 6, 1},
                                                                                               {3, 7, 1},
                                                                                                           {4, 8, 1},
                                                                                                                       {5, 9, 1}
        }, TNL::Matrices::SymmetricMatrixEncoding::SparseMixed );
   // clang-format on

   ASSERT_TRUE( TNL::Graphs::isTree( graph ) );
}

TYPED_TEST( GraphTest, test_isTree_not_tree )
{
   using GraphType = typename TestFixture::GraphType;

   // Create a sample graph.
   // clang-format off
   GraphType graph(
        10, // number of graph nodes
        {   // definition fo graph edges
                       {0, 1, 1},    {0, 2, 1},
                                                {1, 3, 1},  {1, 4, 1},
                                                                        {2, 5, 1},  {2, 6, 1},
                                                                                               {3, 7, 1},
                                                                                                           {4, 8, 1},
            { 5, 0, 1 },                                                                                              {5, 9, 1}
        }, TNL::Matrices::SymmetricMatrixEncoding::SparseMixed );
   // clang-format on

   ASSERT_FALSE( TNL::Graphs::isTree( graph ) );

   // Create another sample graph.
   // clang-format off
   GraphType graph2(
        10, // number of graph nodes
        {   // definition of graph edges
                       {0, 1, 1},    {0, 2, 1},
                                                {1, 3, 1},  {1, 4, 1},
                                                                        {2, 5, 1},  {2, 6, 1},
                                                                                               {3, 7, 1},
                                                                                                           {4, 8, 1},
            { 5, 0, 1 }
        }, TNL::Matrices::SymmetricMatrixEncoding::SparseMixed );
   // clang-format on

   ASSERT_FALSE( TNL::Graphs::isTree( graph2 ) );
}

TYPED_TEST( GraphTest, test_large_tree )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType tree( 29,
                   { { 3, 18, 1.0 },  { 0, 11, 1.0 },  { 4, 19, 1.0 },  { 2, 28, 1.0 },  { 1, 5, 1.0 },   { 8, 12, 1.0 },
                     { 8, 26, 1.0 },  { 11, 28, 1.0 }, { 16, 23, 1.0 }, { 17, 23, 1.0 }, { 13, 19, 1.0 }, { 15, 25, 1.0 },
                     { 18, 25, 1.0 }, { 25, 20, 1.0 }, { 3, 10, 2.0 },  { 4, 12, 2.0 },  { 1, 25, 2.0 },  { 7, 19, 2.0 },
                     { 10, 12, 2.0 }, { 10, 23, 2.0 }, { 14, 18, 2.0 }, { 27, 28, 2.0 }, { 24, 28, 2.0 }, { 0, 22, 3.0 },
                     { 6, 11, 3.0 },  { 9, 17, 3.0 },  { 21, 23, 3.0 }, { 8, 27, 4.0 } },
                   TNL::Matrices::SymmetricMatrixEncoding::SparseMixed );

   ASSERT_TRUE( TNL::Graphs::isTree( tree ) );
}

TYPED_TEST( GraphTest, test_small_forest )
{
   using GraphType = typename TestFixture::GraphType;

   GraphType graph( 5, { { 0, 3, 1.0 }, { 0, 4, 1.0 } }, TNL::Matrices::SymmetricMatrixEncoding::SparseMixed );

   ASSERT_FALSE( TNL::Graphs::isTree( graph ) );
   ASSERT_TRUE( TNL::Graphs::isForest( graph ) );
}

#include "../main.h"
