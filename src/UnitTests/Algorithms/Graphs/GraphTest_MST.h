#include <iostream>
#include <cstdint>

#include <TNL/Algorithms/Graphs/minimumSpanningTree.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Containers/StaticVector.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class GraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using GraphTestTypes = ::testing::Types
<
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int >//,
   //TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >
#ifdef __CUDACC__
   ,TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >
#endif
>;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

TYPED_TEST( GraphTest, test_MST_small )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using EdgeType = TNL::Algorithms::Graphs::Edge< RealType, IndexType >;
   using EdgeArray = TNL::Containers::Array< EdgeType >;

   // Create a TNL::Matrices::SparseMatrix for a sample graph.
   MatrixType matrix(
        6, // number of matrix rows
        6, // number of matrix columns
        {  // matrix elements definition
                        { 0, 1, 1.0 },  { 0, 2, 3.0 },
         { 1, 0, 1.0 },                 { 1, 2, 2.0 },
         { 2, 0, 3.0 }, { 2, 1, 2.0 },
                                                                    { 3, 4, 4.0 }, { 3, 5, 5.0 },
                                                      { 4, 3, 4.0 },               { 4, 5, 3.0 },
                                                      { 5, 3, 5.0 }, { 5, 4, 3.0 }
        });

   MatrixType expected_tree( 6, 6,
      { { 0, 1, 1.0 },
        { 1, 2, 2.0 },
        { 4, 5, 3.0 },
        { 3, 4, 4.0 } }
   );

   MatrixType minimum_tree;
   TNL::Algorithms::Graphs::minimumSpanningTree( matrix, minimum_tree );
   ASSERT_EQ(minimum_tree, expected_tree );
}

TYPED_TEST( GraphTest, test_MST_medium )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using EdgeType = TNL::Algorithms::Graphs::Edge< RealType, IndexType >;
   using EdgeArray = TNL::Containers::Array< EdgeType >;

   // Create a TNL::Matrices::SparseMatrix for a sample graph.
   MatrixType matrix(
        9, // number of matrix rows
        9, // number of matrix columns
        {  // matrix elements definition
                          { 0, 1, 4.0 },                                                                                { 0, 7,  8.0 },
           { 1, 0, 4.0 },                 { 1, 2, 8.0 },                                                                { 1, 7, 11.0 },
                          { 2, 1, 8.0 },                 { 2, 3,  7.0 },                 { 2, 5,  4.0 },                                 { 2, 8, 2.0 },
                                          { 3, 2, 7.0 },                 { 3, 4, 9.0 },  { 3, 5, 14.0 },
                                                         { 4, 3,  9.0 },                 { 4, 5, 10.0 },
                                          { 5, 2, 4.0 }, { 5, 3, 14.0 }, { 5, 4, 10.0 },                 { 5, 6, 2.0 },
                                                                                         { 6, 5,  2.0 },                { 6, 7,  1.0 },  { 6, 8, 6.0 },
           { 7, 0, 8.0 }, { 7, 1, 8.0 },                                                                { 7, 6, 1.0 },                   { 7, 8, 7.0 },
                                          { 8, 2, 2.0 },                                                { 8, 6, 6.0 },  { 8, 7,  7.0 }
         } );

   MatrixType expectedTree( 9, 9,
      {                   { 0, 1, 4.0 },                                                                             { 0, 7, 8.0 },

                                                         { 2, 3, 7.0 },                  { 2, 5, 4.0 },                             { 2, 8, 2.0 },
        { 3, 4, 9.0 },

        { 5, 6, 2.0 },

        { 6, 7, 1.0 }

      } );

   MatrixType minimum_tree;
   TNL::Algorithms::Graphs::minimumSpanningTree( matrix, minimum_tree );
   minimum_tree.sortColumnIndexes();
   const auto& v1 = minimum_tree.getValues();
   const auto& v2 = expectedTree.getValues();
   std::cout << v1 << std::endl;
   std::cout << "minimum tree sum = " << sum(  max( v1, 0 ) ) << std::endl;
   std::cout << "expected tree sum = " << sum(  max( v2, 0 ) ) << std::endl;
   ASSERT_EQ(minimum_tree, expectedTree );
}

TYPED_TEST( GraphTest, test_MST_large )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using EdgeType = TNL::Algorithms::Graphs::Edge< RealType, IndexType >;
   using EdgeArray = TNL::Containers::Array< EdgeType >;

   // Create a TNL::Matrices::SparseMatrix for a sample graph.
   MatrixType matrix(
        20, // number of matrix rows
        20, // number of matrix columns
        {  // matrix elements definition
         {  0,  1,  3.0 }, {  0,  5, 12.0 },
         {  1,  2,  7.0 }, {  1,  6, 11.0 },
         {  2,  3,  4.0 }, {  2,  7,  8.0 }, { 2, 6, 2.0 },
         {  3,  4,  9.0 }, {  3,  7,  3.0 }, { 3, 8, 6.0 },
         {  4,  8,  1.0 }, {  4,  9, 14.0 },
         {  5,  6,  5.0 }, {  5, 10, 15.0 },
         {  6,  7, 10.0 }, {  6, 11,  6.0 },
         {  7, 12, 13.0 },
         {  8,  9,  6.0 }, {  8, 13,  7.0 },
         {  9, 14,  3.0 },
         { 10, 11,  9.0 }, { 10, 15,  8.0 },
         { 11, 12,  5.0 }, { 11, 16,  7.0 },
         { 12, 13,  8.0 }, { 12, 17,  4.0 },
         { 13, 18, 12.0 },
         { 14, 19, 10.0 },
         { 15, 16, 11.0 },
         { 16, 17,  2.0 },
         { 17, 18,  3.0 },
         { 18, 19,  9.0 }
        } );

   MatrixType expected_tree( 20, 20,
      { {  0,  1, 3.0 },
        {  1,  2, 7.0 },
        {  2,  6, 2.0 }, {  2,  3, 4.0 },
        {  3,  7, 3.0 }, {  3,  8, 6.0 },
        {  4,  8, 1.0 },
        {  5,  6, 5.0 },
        {  6, 11, 6.0 },
        {  8,  9, 6.0 }, {  8, 13, 7.0 },
        {  9, 14, 3.0 },
        { 10, 15, 8.0 }, { 10, 11, 9.0 },
        { 11, 12, 5.0 },
        { 12, 17, 4.0 },
        { 16, 17, 2.0 },
        { 17, 18, 3.0 },
        { 18, 19, 9.0 }
      } );

   //TNL::Containers::Array< EdgeType, DeviceType, IndexType > edges;
   MatrixType minimum_tree;
   TNL::Algorithms::Graphs::minimumSpanningTree( matrix, minimum_tree );
   minimum_tree.sortColumnIndexes();
   ASSERT_EQ (minimum_tree, expected_tree );
}

#endif

#include "../../main.h"
