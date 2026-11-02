#pragma once

#include <map>
#include <utility>
#include <vector>

#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseDenseOperations.h>

#include <gtest/gtest.h>

// The product is computed in tiles of at most this number of columns of B, the tests
// cover several tiles and a smaller last tile.
constexpr int testMaxTileSize = 4;

// Tests C = alpha * A * B + beta * C for a matrix A with empty, short and long rows and
// dense matrices B and C with the given organization. All values are small integers, so the
// results are exact for all tested types.
template< typename Matrix, TNL::Algorithms::Segments::ElementsOrganization Organization >
void
test_SparseDenseMatrixProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DenseMatrixType = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType, Organization >;
   using HostDenseMatrixType = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Host, IndexType, Organization >;

   // 257 is a prime, so the column indexes of every row are distinct
   const IndexType rows = 300;
   const IndexType columns = 257;
   const std::vector< IndexType > rowLengths{ 0, 1, 7, 33, 100, 256, 2, 0, 65, 3 };
   std::map< std::pair< IndexType, IndexType >, RealType > elements;
   for( IndexType row = 0; row < rows; row++ ) {
      const IndexType length = rowLengths[ row % rowLengths.size() ];
      for( IndexType localIdx = 0; localIdx < length; localIdx++ )
         elements[ { row, ( row + 3 * localIdx ) % columns } ] = RealType( row % 5 + localIdx % 3 + 1 );
   }
   const Matrix A( rows, columns, elements );

   for( const IndexType k : { 1, 3, 4, 5, 9 } ) {
      HostDenseMatrixType hostB( columns, k );
      HostDenseMatrixType hostC0( rows, k );
      for( IndexType i = 0; i < columns; i++ )
         for( IndexType j = 0; j < k; j++ )
            hostB.setElement( i, j, RealType( i % 7 + j + 1 ) );
      for( IndexType i = 0; i < rows; i++ )
         for( IndexType j = 0; j < k; j++ )
            hostC0.setElement( i, j, RealType( i % 11 - j ) );
      DenseMatrixType B;
      B = hostB;

      for( const auto& [ alpha, beta ] : { std::pair< int, int >{ 1, 0 }, std::pair< int, int >{ 2, -1 } } ) {
         // the expected result
         HostDenseMatrixType expected( hostC0 );
         for( IndexType i = 0; i < rows; i++ )
            for( IndexType j = 0; j < k; j++ )
               expected.setElement( i, j, RealType( beta ) * hostC0.getElement( i, j ) );
         for( const auto& [ position, value ] : elements )
            for( IndexType j = 0; j < k; j++ )
               expected.addElement(
                  position.first, j, RealType( alpha ) * value * hostB.getElement( position.second, j ), RealType( 1 ) );

         for( const auto& [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( A.getSegments() ) )
         {
            SCOPED_TRACE(
               tag + ", k = " + std::to_string( k ) + ", alpha = " + std::to_string( alpha )
               + ", beta = " + std::to_string( beta ) );
            DenseMatrixType C;
            C = hostC0;
            TNL::Matrices::sparseDenseMatrixProduct< testMaxTileSize >(
               A, B, C, RealType( alpha ), RealType( beta ), launch_config );
            HostDenseMatrixType hostC;
            hostC = C;
            for( IndexType i = 0; i < rows; i++ )
               for( IndexType j = 0; j < k; j++ )
                  EXPECT_EQ( hostC.getElement( i, j ), expected.getElement( i, j ) ) << "i = " << i << ", j = " << j;
         }
      }
   }
}

template< typename Matrix >
void
test_SparseDenseMatrixProduct_wrongDimensions()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DenseMatrixType = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType >;

   const Matrix A( 5, 4 );
   DenseMatrixType C( 5, 3 );
   EXPECT_THROW( TNL::Matrices::sparseDenseMatrixProduct( A, DenseMatrixType( 3, 3 ), C ), std::invalid_argument );
   EXPECT_THROW( TNL::Matrices::sparseDenseMatrixProduct( A, DenseMatrixType( 4, 2 ), C ), std::invalid_argument );
   DenseMatrixType C2( 6, 3 );
   EXPECT_THROW( TNL::Matrices::sparseDenseMatrixProduct( A, DenseMatrixType( 4, 3 ), C2 ), std::invalid_argument );
}
