// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <string>

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Containers/VectorView.h>

namespace TNL::Matrices::Factorization::QR {

template< typename MatrixType >
void
GramSchmidt( const MatrixType& A, MatrixType& Q, MatrixType& R )
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
   using VectorViewType = Containers::VectorView< RealType, DeviceType, IndexType >;

   // initialize the matrices
   Q = A;
   R.setDimensions( A.getColumns(), A.getColumns() );
   R.setValue( 0 );

   // compute the first orthogonal vector
   VectorViewType q_0;
   q_0.bind( &Q( 0, 0 ), Q.getRows() );
   const RealType norm = TNL::l2Norm( q_0 );
   q_0 /= norm;
   R( 0, 0 ) = norm;

   // compute the remaining orthogonal vectors using modified Gram-Schmidt process
   for( IndexType i = 1; i < A.getColumns(); i++ ) {
      VectorViewType q_i;
      q_i.bind( &Q( 0, i ), Q.getRows() );

      for( IndexType j = 0; j < i; j++ ) {
         VectorViewType q_j;
         q_j.bind( &Q( 0, j ), Q.getRows() );
         const RealType r_ji = TNL::dot( q_j, q_i );
         R( j, i ) = r_ji;
         q_i -= r_ji * q_j;
      }

      const RealType norm = TNL::l2Norm( q_i );
      q_i /= norm;
      R( i, i ) = norm;
   }
}

template< typename MatrixType >
std::pair< MatrixType, MatrixType >
GramSchmidt( const MatrixType& A )
{
   MatrixType Q;
   MatrixType R;
   GramSchmidt( A, Q, R );
   return { Q, R };
}

}  // namespace TNL::Matrices::Factorization::QR
