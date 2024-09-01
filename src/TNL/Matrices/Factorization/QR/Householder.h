// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <string>

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

namespace TNL::Matrices::Factorization::QR {

template< typename MatrixType >
void
Householder( const MatrixType& A, MatrixType& Q, MatrixType& R )
{
   static_assert( MatrixType::getOrganization() == Algorithms::Segments::ColumnMajorOrder,
                  "The input matrix must have the column-major order." );

   if( A.getColumns() > A.getRows() )
      throw std::invalid_argument( "The input matrix must be square or have more rows than columns. It has "
                                   + std::to_string( A.getRows() ) + " rows and " + std::to_string( A.getColumns() )
                                   + " columns." );

   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using VectorViewType = Containers::VectorView< RealType, DeviceType, IndexType >;

   // initialize Q to the identity matrix
   Q.setDimensions( A.getRows(), A.getColumns() );
   Q.setValue( 0 );
   for( IndexType i = 0; i < Q.getRows(); i++ )
      Q( i, i ) = 1;

   // initialize R to the matrix A
   R = A;

   // storage for the Householder reflection vector
   VectorType U( R.getRows() );

   for( IndexType i = 0; i < R.getColumns(); i++ ) {
      // bind truncated column of the matrix R
      VectorViewType r_i;
      r_i.bind( &R( i, i ), R.getRows() - i );

      // bind truncated part of the vector U
      VectorViewType u_i;
      u_i.bind( U.getData(), R.getRows() - i );

      // compute the norm of the truncated vector
      const RealType norm = TNL::l2Norm( r_i );

      // set the Householder reflection vector u_i and zero out r_i below its first element
      u_i = r_i;
      r_i = 0;
      if( u_i[ 0 ] > 0 ) {
         u_i[ 0 ] = u_i[ 0 ] + norm;
         r_i[ 0 ] = -norm;
      }
      else {
         u_i[ 0 ] = u_i[ 0 ] - norm;
         r_i[ 0 ] = norm;
      }
      u_i = u_i / TNL::l2Norm( u_i );

      for( IndexType j = i + 1; j < R.getColumns(); j++ ) {
         // bind truncated column of the matrix R
         VectorViewType r_j;
         r_j.bind( &R( i, j ), R.getRows() - i );
         // apply the Householder reflection
         r_j -= 2 * TNL::dot( u_i, r_j ) * u_i;
      }

      for( IndexType j = 0; j < Q.getColumns(); j++ ) {
         // bind truncated column of the matrix Q
         VectorViewType q_j;
         q_j.bind( &Q( i, j ), Q.getRows() - i );
         // apply the Householder reflection
         q_j -= 2 * TNL::dot( u_i, q_j ) * u_i;
      }
   }

   MatrixType Q2;
   Q2.setDimensions( A.getRows(), A.getColumns() );
   Q2.getTransposition( Q );
   std::swap( Q2, Q );
}

template< typename MatrixType >
std::pair< MatrixType, MatrixType >
Householder( MatrixType& A )
{
   MatrixType Q;
   MatrixType R;
   Householder( A, Q, R );
   return { Q, R };
}

}  // namespace TNL::Matrices::Factorization::QR
