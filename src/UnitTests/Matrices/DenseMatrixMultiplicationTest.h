#include <gtest/gtest.h>
#include <functional>
#include <cmath>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <iostream>
#include <sstream>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Matrices/DenseMatrix.h>

template< typename Matrix >
class DenseMatrixMultiplicationTest : public ::testing::Test
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

TYPED_TEST_SUITE( DenseMatrixMultiplicationTest, MatrixTypes );

TYPED_TEST( DenseMatrixMultiplicationTest, IdentityProduct )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 1, 4, 5, 6, 3 }, { 8, 1, 0, 4, 5 }, { 5, 0, 1, 0, 5 }, { 4, 5, 2, 1, 7 }, { 7, 8, 0, 9, 9 } };

   DenseMatrixType matrix2{ { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1 } };

   DenseMatrixType resultMatrix;

   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == matrix1;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTest, ZeroMatrixProduct )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   DenseMatrixType matrix2{
      { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }
   };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 } };

   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTest, EmptyMatrices )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   // Both matrices are empty
   DenseMatrixType matrix1{};
   DenseMatrixType matrix2{};

   DenseMatrixType resultMatrix;

   // Expected result is also an empty matrix
   DenseMatrixType checkMatrix{};

   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}
TYPED_TEST( DenseMatrixMultiplicationTest, SingleElementMatrices )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 5 } };
   DenseMatrixType matrix2{ { 4 } };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 20 } };  // 5 * 4 = 20

   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTest, VectorMatrixMultiplication )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   // Vector treated as 1xN matrix
   DenseMatrixType vector{ { 1, 2 } };
   DenseMatrixType matrix{ { 3, 4 }, { 5, 6 } };

   DenseMatrixType resultMatrix;

   // Expected: [1*3 + 2*5, 1*4 + 2*6] = [13, 16]
   DenseMatrixType checkMatrix{ { 13, 16 } };

   resultMatrix.getMatrixProduct( vector, matrix, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTest, VectorDotProduct )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   // Both vectors treated as 1xN and Nx1 matrices
   DenseMatrixType vector1{ { 1, 2, 3 } };
   DenseMatrixType vector2{ { 4 }, { 5 }, { 6 } };

   DenseMatrixType resultMatrix;

   // Expected dot product: 1*4 + 2*5 + 3*6 = 32
   DenseMatrixType checkMatrix{ { 32 } };

   resultMatrix.getMatrixProduct( vector1, vector2, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTest, NormalProduct1 )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   DenseMatrixType matrix2{
      { 4, 3, 2, 6, 7, 8, 4 }, { 3, 4, 2, 8, 6, 1, 5 }, { 1, 4, 7, 9, 5, 2, 6 }, { 2, 3, 8, 0, 4, 0, 7 }
   };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 51, 76, 95, 141, 118, 60, 113 },
                                { 47, 55, 60, 94, 94, 57, 82 },
                                { 64, 77, 105, 101, 125, 57, 128 } };

   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTest, NormalProduct2 )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 167, 209, 343 }, { 765, 332, 424 }, { 542, 52, 51 },   { 214, 211, 78 },  { 421, 12, 87 },
                            { 123, 732, 8 },   { 532, 677, 898 }, { 525, 523, 678 }, { 534, 345, 645 }, { 542, 456, 432 },
                            { 321, 23, 525 },  { 324, 872, 425 }, { 567, 987, 568 } };

   DenseMatrixType matrix2{ { 134, 567, 454, 323, 234, 342, 865, 543, 146, 642 },
                            { 543, 589, 577, 432, 242, 643, 533, 477, 642, 123 },
                            { 872, 876, 124, 456, 245, 634, 789, 754, 907, 345 } };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 434961, 518258, 238943, 300637, 173691, 408963, 526479, 448996, 469661, 251256 },
                                { 652514, 1000727, 591450, 583863, 363234, 743922, 1173217, 893455, 709402, 678246 },
                                { 145336, 382618, 282396, 220786, 151907, 251134, 536785, 357564, 158773, 371955 },
                                { 211265, 313945, 228575, 195842, 120248, 258313, 359115, 275661, 237452, 190251 },
                                { 138794, 321987, 208846, 180839, 122733, 206856, 439204, 299925, 148079, 301773 },
                                { 420934, 507897, 479198, 359601, 207886, 517814, 502863, 421985, 495158, 171762 },
                                { 1221955, 1487045, 743509, 873788, 508332, 1186587, 1529543, 1288897, 1326792, 734625 },
                                { 945555, 1199650, 624193, 704679, 415526, 945691, 1267826, 1045758, 1027362, 635289 },
                                { 821331, 1071003, 521481, 615642, 366471, 813393, 1154700, 940857, 884469, 607788 },
                                { 696940, 954330, 562748, 569050, 343020, 752460, 1052726, 837546, 763708, 553092 },
                                { 513303, 655454, 224105, 353019, 209305, 457421, 704149, 581124, 537807, 390036 },
                                { 887512, 1069616, 702940, 675156, 390965, 940954, 1080361, 912326, 992603, 461889 },
                                { 1107215, 1400400, 897349, 868533, 510692, 1188667, 1464678, 1206952, 1231612, 681375 } };

   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTest, NormalProductWrong )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   DenseMatrixType matrix2{
      { 4, 3, 2, 6, 7, 8, 4 }, { 3, 4, 2, 8, 6, 1, 5 }, { 1, 4, 7, 9, 5, 2, 6 }, { 2, 3, 8, 0, 4, 0, 7 }
   };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 52, 76, 95, 141, 118, 60, 113 },
                                { 47, 55, 60, 94, 94, 57, 82 },
                                { 64, 77, 105, 101, 125, 57, 128 } };

   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, false );
}

TYPED_TEST( DenseMatrixMultiplicationTest, LargeMatrices )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1;
   matrix1.setDimensions( 50, 70 );

   DenseMatrixType matrix2;
   matrix2.setDimensions( 70, 40 );

   // Fill the matrices
   const double h_x = 1.0 / 100;
   const double h_y = 1.0 / 100;

   for( int i = 0; i < matrix1.getRows(); i++ ) {
      for( int j = 0; j < matrix1.getColumns(); j++ ) {
         double value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * j );
         matrix1.setElement( i, j, value );
      }
   }

   for( int i = 0; i < matrix2.getRows(); i++ ) {
      for( int j = 0; j < matrix2.getColumns(); j++ ) {
         double value = std::sin( 3 * M_PI * h_x * i ) + std::cos( 3 * M_PI * h_y * j );
         matrix2.setElement( i, j, value );
      }
   }

   DenseMatrixType resultMatrix;
   resultMatrix.setDimensions( 50, 40 );

   DenseMatrixType checkMatrix;
   checkMatrix.setDimensions( 50, 40 );

   // Calculate the product of matrix1 and matrix2 manually
   for( int i = 0; i < matrix1.getRows(); i++ ) {
      for( int j = 0; j < matrix2.getColumns(); j++ ) {
         double sum = 0;
         for( int k = 0; k < matrix1.getColumns(); k++ ) {
            sum += matrix1.getElement( i, k ) * matrix2.getElement( k, j );
         }
         checkMatrix.setElement( i, j, sum );
      }
   }

   resultMatrix.getMatrixProduct( matrix1, matrix2 );

   auto tolerance = 1e-3;

   auto diff = resultMatrix.getValues() - checkMatrix.getValues();

   bool check = true;

   if( TNL::maxNorm( abs( diff ) ) > tolerance ) {
      check = false;
   }
   ASSERT_EQ( check, true );
}

#include "../main.h"
