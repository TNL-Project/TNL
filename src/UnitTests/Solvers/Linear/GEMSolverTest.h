#include <iostream>
#include <functional>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/Linear/GEM.h>

#include <type_traits>

#include <gtest/gtest.h>

using Dense_host_float = TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int >;
using Dense_host_int = TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int >;

using Dense_cuda_float = TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int >;
using Dense_cuda_int = TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int >;

// test fixture for typed tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using MatrixTypes =
   ::testing::Types< TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
                     TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder > /*,
                      TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder
                      >*/
#if defined( __CUDACC__ )
//,
//TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder
//>, TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int,
//TNL::Algorithms::Segments::ColumnMajorOrder >, TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda,
//long, TNL::Algorithms::Segments::ColumnMajorOrder >,
// TNL::Matrices::DenseMatrix< TNL::Arithmetics::Complex<float>,  TNL::Devices::Cuda, long,
// TNL::Algorithms::Segments::ColumnMajorOrder >,
//   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::RowMajorOrder >
#elif defined( __HIP__ )
                     ,
                     TNL::Matrices::DenseMatrix< float, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
                     TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ColumnMajorOrder >,
                     TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int, TNL::Algorithms::Segments::RowMajorOrder >
#endif
                     >;

template< typename Matrix >
void
test_diagonalMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 5x5 dense matrix:
    *
    *    / 1 . . . . \
    *    | . 2 . . . |
    *    | . . 3 . . |
    *    | . . . 4 . |
    *    \ . . . . 5 /
    */
   const IndexType size = 4;

   Matrix matrix( size, size );
   matrix.getValues() = 0;

   IndexType value = 1;
   for( IndexType i = 0; i < size; i++ )
      matrix.setElement( i, i, value++ );

   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix.vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem( matrix, b );
   gem.solveWithPivoting( y, 10 );

   EXPECT_NEAR( maxNorm( x - y ), 0.0, 1.0e-6 );
}

template< typename Matrix >
void
test_upperTriangularMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 5x5 dense matrix:
    *
    *    / 1 2 3 4 5 \
    *    | . 1 2 3 4 |
    *    | . . 1 2 3 |
    *    | . . . 1 2 |
    *    \ . . . . 1 /
    */
   const IndexType size = 4;

   Matrix matrix( size, size );
   matrix.getValues() = 0;

   for( IndexType i = 0; i < size; i++ )
      for( IndexType j = 0; j < size; i++ )
         if( j >= i )
            matrix.setElement( i, j, j - i );

   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix.vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem( matrix, b );
   gem.solveWithPivoting( y, 10 );

   EXPECT_NEAR( maxNorm( x - y ), 0.0, 1.0e-6 );
}

template< typename Matrix >
void
test_smallMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 4x4 dense matrix:
    *
    *    /  1  2  3  4 \
    *    |  5  6  7  8 |
    *    |  9 10 11 12 |
    *    \ 13 14 15 16 /
    */
   const IndexType size = 4;

   Matrix matrix( size, size );

   IndexType value = 1;
   for( IndexType i = 0; i < size; i++ )
      for( IndexType j = 0; j < size; j++ )
         matrix.setElement( i, j, value++ );

   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix.vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem( matrix, b );
   gem.solveWithPivoting( y, 10 );

   std::cout << " y = " << y << std::endl;
}

TYPED_TEST_SUITE( MatrixTest, MatrixTypes );

TYPED_TEST( MatrixTest, diagonalMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_diagonalMatrix< MatrixType >();
}

TYPED_TEST( MatrixTest, upperTriangularMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_upperTriangularMatrix< MatrixType >();
}

/*TYPED_TEST( MatrixTest, smallMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_smallMatrix< MatrixType >();
}*/

#include "../../main.h"
