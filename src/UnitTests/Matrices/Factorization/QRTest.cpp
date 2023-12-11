#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/Factorization/QR/GramSchmidt.h>
#include <TNL/Matrices/Factorization/QR/Householder.h>
#include <TNL/Matrices/Factorization/QR/Givens.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Matrices;
using namespace TNL::Matrices::Factorization::QR;

template< typename MatrixType >
void
test_orthogonal( const MatrixType matrix )
{
   static_assert( MatrixType::getOrganization() == Algorithms::Segments::ColumnMajorOrder,
                  "The input matrix must have the column-major order." );

   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorViewType = Containers::VectorView< std::add_const_t< RealType >, DeviceType, IndexType >;

   const RealType epsilon = 10 * std::numeric_limits< RealType >::epsilon();

   for( IndexType i = 0; i < matrix.getColumns(); i++ ) {
      VectorViewType q_i;
      q_i.bind( &matrix( 0, i ), matrix.getRows() );
      EXPECT_NEAR( TNL::dot( q_i, q_i ), 1, epsilon ) << "i=" << i;

      for( IndexType j = i + 1; j < matrix.getColumns(); j++ ) {
         VectorViewType q_j;
         q_j.bind( &matrix( 0, j ), matrix.getRows() );
         EXPECT_NEAR( TNL::dot( q_i, q_j ), 0, epsilon ) << "i=" << i;
      }
   }
}

template< typename MatrixType >
void
test_upper_triangular( const MatrixType matrix )
{
   using IndexType = typename MatrixType::IndexType;

   for( IndexType i = 0; i < matrix.getRows(); i++ ) {
      for( IndexType j = 0; j < i; j++ ) {
         EXPECT_EQ( matrix( i, j ), 0 ) << "i=" << i << ", j=" << j;
      }
   }
}

template< typename MatrixType >
void
test_QR_factorization( const MatrixType A, const MatrixType Q, const MatrixType R )
{
   using RealType = typename MatrixType::RealType;

   // check dimensions
   EXPECT_EQ( Q.getRows(), A.getRows() );
   EXPECT_EQ( Q.getColumns(), A.getColumns() );
   EXPECT_EQ( R.getRows(), A.getColumns() );
   EXPECT_EQ( R.getColumns(), A.getColumns() );

   // check that R is upper triangular
   test_upper_triangular( R );

   // check that Q is orthogonal
   test_orthogonal( Q );

   // compute the product of Q and R
   MatrixType B;
   B.getMatrixProduct( Q, R );

   // compute the error matrix  E = A - B
   MatrixType E;
   E = A;
   E.addMatrix( B, -1.0 );

   // compute the Frobenius norm of the error
   const RealType norm2 = TNL::dot( E.getValues(), E.getValues() );
   const RealType epsilon = std::numeric_limits< RealType >::epsilon();
   EXPECT_NEAR( norm2, 0.0, epsilon ) << "A=\n" << A << "\nQR=\n" << B;
}

TEST( QRTest, GramSchmidt )
{
   using RealType = float;
   using MatrixType = DenseMatrix< RealType, Devices::Host, int, Algorithms::Segments::ColumnMajorOrder >;

   // input matrix for the test
   const MatrixType A = { { 1.0, 2.0 }, { 3.0, 4.0 } };

   // compute the QR factorization
   MatrixType Q;
   MatrixType R;
   GramSchmidt( A, Q, R );

   // verify the QR factorization
   test_QR_factorization( A, Q, R );
}

TEST( QRTest, Householder )
{
   using RealType = float;
   using MatrixType = DenseMatrix< RealType, Devices::Host, int, Algorithms::Segments::ColumnMajorOrder >;

   // input matrix for the test
   const MatrixType A = { { 1.0, 2.0 }, { 3.0, 4.0 } };

   // compute the QR factorization
   MatrixType Q;
   MatrixType R;
   Householder( A, Q, R );

   // verify the QR factorization
   test_QR_factorization( A, Q, R );
}

TEST( QRTest, Givens )
{
   using RealType = float;
   using MatrixType = DenseMatrix< RealType, Devices::Host, int, Algorithms::Segments::ColumnMajorOrder >;

   // input matrix for the test
   const MatrixType A = { { 1.0, 2.0 }, { 3.0, 4.0 } };

   // compute the QR factorization
   MatrixType Q;
   MatrixType R;
   Givens( A, Q, R );

   // verify the QR factorization
   test_QR_factorization( A, Q, R );
}

#include "../../main.h"
