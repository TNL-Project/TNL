#include <gtest/gtest.h>
#include <functional>
#include <iostream>
#include <sstream>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Matrices/DenseMatrix.h>

template< typename Matrix >
class DenseMatrixTranspositionTest : public ::testing::Test
{
public:
   using DenseMatrixType = Matrix;
};

// Define MatrixTypes for the test suite
using MatrixTypes = ::testing::Types<
   TNL::Matrices::
      DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >,
   TNL::Matrices::
      DenseMatrix< float, TNL::Devices::Host, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::
      DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >,
   TNL::Matrices::
      DenseMatrix< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >

#elif defined( __HIP__ )
   ,
   TNL::Matrices::
      DenseMatrix< double, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >,
   TNL::Matrices::
      DenseMatrix< float, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >
#endif
   >;

TYPED_TEST_SUITE( DenseMatrixTranspositionTest, MatrixTypes );

TYPED_TEST( DenseMatrixTranspositionTest, EmptyMatrix )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{};         // Empty matrix
   DenseMatrixType InPlaceMatrix{};  //Empty matrix for the InPlace Transposition

   DenseMatrixType resultMatrix;

   resultMatrix.getTransposition( matrix );
   InPlaceMatrix.getInPlaceTransposition();

   // Expect the result to be an empty matrix as well
   DenseMatrixType checkMatrix{};

   bool check = ( resultMatrix == checkMatrix ) && ( InPlaceMatrix == checkMatrix );
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixTranspositionTest, SingleElementMatrix )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 5 } };         // 1x1 matrix
   DenseMatrixType InPlaceMatrix{ { 5 } };  // 1x1 matrix for the InPlace Transposition

   DenseMatrixType resultMatrix;

   resultMatrix.getTransposition( matrix );

   // Check matrix is the same as the input matrix
   DenseMatrixType checkMatrix{ { 5 } };

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixTranspositionTest, SingleRowMatrix )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 1, 2, 3, 4 } };  // Single row matrix

   DenseMatrixType resultMatrix;

   resultMatrix.getTransposition( matrix );

   // Expected matrix is a single column matrix with the elements of the original row
   DenseMatrixType checkMatrix{ { 1 }, { 2 }, { 3 }, { 4 } };

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixTranspositionTest, SingleColumnMatrix )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 1 }, { 2 }, { 3 }, { 4 } };  // Single column matrix

   DenseMatrixType resultMatrix;

   resultMatrix.getTransposition( matrix );

   // Expected matrix is a single row with the elements of the original column
   DenseMatrixType checkMatrix{ { 1, 2, 3, 4 } };

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixTranspositionTest, SquareMatrix )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } };
   DenseMatrixType InPlaceMatrix{ { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } };

   DenseMatrixType resultMatrix;
   resultMatrix.getTransposition( matrix );
   InPlaceMatrix.getInPlaceTransposition();

   DenseMatrixType checkMatrix{ { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 3, 3, 3, 3 }, { 4, 4, 4, 4 } };

   bool check = ( resultMatrix == checkMatrix ) && ( InPlaceMatrix == checkMatrix );
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixTranspositionTest, Transposition1 )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 5, 6, 6 }, { 6, 5, 7 }, { 7, 2, 1 }, { 3, 3, 9 } };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   resultMatrix.getTransposition( matrix );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixTranspositionTest, TranspositionWrong )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 5, 6, 6 }, { 6, 5, 7 }, { 7, 2, 1 }, { 3, 3, 9 } };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 1, 1, 1, 1 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   resultMatrix.getTransposition( matrix );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, false );
}

TYPED_TEST( DenseMatrixTranspositionTest, Transposition2 )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 4, 3, 1, 2 }, { 3, 4, 4, 3 }, { 2, 2, 7, 8 }, { 6, 8, 9, 0 },
                           { 7, 6, 5, 4 }, { 8, 1, 2, 0 }, { 4, 5, 6, 7 } };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{
      { 4, 3, 2, 6, 7, 8, 4 }, { 3, 4, 2, 8, 6, 1, 5 }, { 1, 4, 7, 9, 5, 2, 6 }, { 2, 3, 8, 0, 4, 0, 7 }
   };

   resultMatrix.getTransposition( matrix );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixTranspositionTest, Transposition3 )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 134, 567, 454, 323, 234, 342, 865, 543, 146, 642 },
                           { 543, 589, 577, 432, 242, 643, 533, 477, 642, 123 },
                           { 872, 876, 124, 456, 245, 634, 789, 754, 907, 345 } };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 134, 543, 872 }, { 567, 589, 876 }, { 454, 577, 124 }, { 323, 432, 456 }, { 234, 242, 245 },
                                { 342, 643, 634 }, { 865, 533, 789 }, { 543, 477, 754 }, { 146, 642, 907 }, { 642, 123, 345 } };

   resultMatrix.getTransposition( matrix );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixTranspositionTest, LargeMatrices )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix( 110, 110 );
   DenseMatrixType InPlaceMatrix( 110, 110 );

   // Fill the matrices
   const double h_x = 1.0 / 100;
   const double h_y = 1.0 / 100;
   for( int i = 0; i < matrix.getRows(); i++ ) {
      for( int j = 0; j < matrix.getColumns(); j++ ) {
         double value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * j );
         matrix.setElement( i, j, value );
         InPlaceMatrix.setElement( i, j, value );
      }
   }

   DenseMatrixType checkMatrix( 110, 110 );

   // Transpose the checkMatrix manually
   for( int i = 0; i < checkMatrix.getRows(); i++ ) {
      for( int j = 0; j < checkMatrix.getColumns(); j++ ) {
         checkMatrix.setElement( i, j, matrix.getElement( j, i ) );
      }
   }

   DenseMatrixType resultMatrix;
   resultMatrix.getTransposition( matrix );

   InPlaceMatrix.getInPlaceTransposition();

   auto tolerance = 1e-3;

   auto diff1 = resultMatrix.getValues() - checkMatrix.getValues();
   auto diff2 = InPlaceMatrix.getValues() - checkMatrix.getValues();

   bool check1 = true;
   bool check2 = true;

   if( TNL::maxNorm( abs( diff1 ) ) > tolerance ) {
      check1 = false;
   }
   if( TNL::maxNorm( abs( diff2 ) ) > tolerance ) {
      check2 = false;
   }
   ASSERT_TRUE( check1 && check2 );
}

#include "../main.h"
