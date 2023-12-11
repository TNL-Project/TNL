// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <string>

#include <TNL/Math.h>
#include <TNL/Containers/VectorView.h>

namespace TNL::Matrices::Factorization::QR {

template< typename MatrixType >
void
Givens( const MatrixType& A, MatrixType& Q, MatrixType& R )
{
   if( A.getColumns() > A.getRows() )
      throw std::invalid_argument( "The input matrix must be square or have more rows than columns. It has "
                                   + std::to_string( A.getRows() ) + " rows and " + std::to_string( A.getColumns() )
                                   + " columns." );

   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;

   // initialize Q to the identity matrix
   Q.setDimensions( A.getRows(), A.getColumns() );
   Q.setValue( 0 );
   for( IndexType i = 0; i < Q.getRows(); i++ )
      Q( i, i ) = 1;

   // initialize R to the matrix A
   R = A;

   for( IndexType i = 0; i < A.getRows(); i++ ) {
      for( IndexType j = i + 1; j < A.getRows(); j++ ) {
         RealType a = R( i, i );
         RealType b = R( j, i );
         RealType c;
         RealType s;
         RealType r;

         // stable computation based on Edward Anderson's algorithm for LAPACK (ensures that r is positive)
         // https://en.wikipedia.org/w/index.php?title=Givens_rotation&oldid=1185759283#Stable_calculation
         if( b == 0 ) {
            c = TNL::sign( a );
            if( c == 0 )
               c = 1;
            s = 0;
            r = std::abs( a );
         }
         else if( a == 0 ) {
            c = 0;
            s = -TNL::sign( b );
            r = std::abs( b );
         }
         else if( std::abs( a ) > std::abs( b ) ) {
            const RealType t = b / a;
            const RealType u = TNL::sign( a ) * std::sqrt( 1 + t * t );
            c = 1 / u;
            s = -c * t;
            r = a * u;
         }
         else {
            const RealType t = a / b;
            const RealType u = TNL::sign( b ) * std::sqrt( 1 + t * t );
            s = -1 / u;
            c = t / u;
            r = b * u;
         }

         // apply the rotation on the current column in R
         R( i, i ) = r;
         R( j, i ) = 0;

         // apply the rotation on the remaining columns in matrix R
         for( IndexType k = i + 1; k < R.getColumns(); k++ ) {
            const RealType r_ik = R( i, k );
            const RealType r_jk = R( j, k );
            R( i, k ) = c * r_ik + -s * r_jk;
            R( j, k ) = s * r_ik + c * r_jk;
         }

         // apply the rotation on matrix Q
         for( IndexType k = 0; k < Q.getRows(); k++ ) {
            const RealType q_ki = Q( k, i );
            const RealType q_kj = Q( k, j );
            Q( k, i ) = c * q_ki - s * q_kj;
            Q( k, j ) = s * q_ki + c * q_kj;
         }
      }
   }
}

template< typename MatrixType >
std::pair< MatrixType, MatrixType >
Givens( const MatrixType& A )
{
   MatrixType Q;
   MatrixType R;
   Givens( A, Q, R );
   return { Q, R };
}

}  // namespace TNL::Matrices::Factorization::QR
