#pragma once

#include <functional>
#include <map>
#include <string>
#include <type_traits>
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
// results are exact for all tested types. The function getProducts( A ) returns pairs of a tag
// and a function product( B, C, alpha, beta ) computing C = alpha * A * B + beta * C.
template< typename Matrix, TNL::Algorithms::Segments::ElementsOrganization Organization, typename GetProducts >
void
test_SparseDenseMatrixProductVariants( GetProducts&& getProducts )
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

   // more than 8 columns cover the automatic selection of sparseDenseMatrixProductLanesOverColumns for CSR
   for( const IndexType k : { 1, 3, 4, 5, 9, 17, 33, 70 } ) {
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

         for( const auto& [ tag, product ] : getProducts( A ) ) {
            SCOPED_TRACE(
               tag + ", k = " + std::to_string( k ) + ", alpha = " + std::to_string( alpha )
               + ", beta = " + std::to_string( beta ) );
            DenseMatrixType C;
            C = hostC0;
            product( B, C, RealType( alpha ), RealType( beta ) );
            HostDenseMatrixType hostC;
            hostC = C;
            for( IndexType i = 0; i < rows; i++ )
               for( IndexType j = 0; j < k; j++ )
                  EXPECT_EQ( hostC.getElement( i, j ), expected.getElement( i, j ) ) << "i = " << i << ", j = " << j;
         }
      }
   }
}

// The type of the functions returned by the getProducts functions of test_SparseDenseMatrixProductVariants.
template< typename Matrix, TNL::Algorithms::Segments::ElementsOrganization Organization >
using SparseDenseMatrixProductFunction = std::function< void(
   const TNL::Matrices::
      DenseMatrix< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, Organization >&,
   TNL::Matrices::
      DenseMatrix< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, Organization >&,
   typename Matrix::RealType,
   typename Matrix::RealType ) >;

// Tests sparseDenseMatrixProduct with all launch configurations of the segments of the matrix.
template< typename Matrix, TNL::Algorithms::Segments::ElementsOrganization Organization >
void
test_SparseDenseMatrixProduct()
{
   using RealType = typename Matrix::RealType;
   using Function = SparseDenseMatrixProductFunction< Matrix, Organization >;
   test_SparseDenseMatrixProductVariants< Matrix, Organization >(
      []( const Matrix& A )
      {
         std::vector< std::pair< std::string, Function > > products;
         for( const auto& [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( A.getSegments() ) )
            products.emplace_back(
               tag,
               [ &A, launch_config = launch_config ]( const auto& B, auto& C, RealType alpha, RealType beta )
               {
                  TNL::Matrices::sparseDenseMatrixProduct< testMaxTileSize >( A, B, C, alpha, beta, launch_config );
               } );
         return products;
      } );
}

// Tests sparseDenseMatrixProductLanesOverColumns with several numbers of lanes and columns per lane,
// so that the tiles of columns are larger than, equal to and smaller than the numbers of columns of B.
// The small maximal row lengths split the longer rows into pieces (on GPUs), with the piece size equal
// to the maximal row length.
template< typename Matrix, TNL::Algorithms::Segments::ElementsOrganization Organization >
void
test_SparseDenseMatrixProductLanesOverColumns()
{
   using RealType = typename Matrix::RealType;
   using Function = SparseDenseMatrixProductFunction< Matrix, Organization >;
   test_SparseDenseMatrixProductVariants< Matrix, Organization >(
      []( const Matrix& A )
      {
         std::vector< std::pair< std::string, Function > > products;
         using IndexType = typename Matrix::IndexType;
         auto add = [ & ]( auto lanes, auto columnsPerLane, IndexType maxRowLength )
         {
            constexpr int Lanes = decltype( lanes )::value;
            constexpr int ColumnsPerLane = decltype( columnsPerLane )::value;
            products.emplace_back(
               "Lanes = " + std::to_string( Lanes ) + ", columns per lane = " + std::to_string( ColumnsPerLane )
                  + ", maximal row length = " + std::to_string( maxRowLength ),
               [ &A, maxRowLength ]( const auto& B, auto& C, RealType alpha, RealType beta )
               {
                  TNL::Matrices::sparseDenseMatrixProductLanesOverColumns< Lanes, ColumnsPerLane >(
                     A, B, C, alpha, beta, maxRowLength );
               } );
         };
         add( std::integral_constant< int, 1 >{}, std::integral_constant< int, 1 >{}, 0 );
         add( std::integral_constant< int, 1 >{}, std::integral_constant< int, 3 >{}, 0 );
         add( std::integral_constant< int, 4 >{}, std::integral_constant< int, 2 >{}, 0 );
         add( std::integral_constant< int, 32 >{}, std::integral_constant< int, 1 >{}, 0 );
         add( std::integral_constant< int, 1 >{}, std::integral_constant< int, 3 >{}, 1 );
         add( std::integral_constant< int, 4 >{}, std::integral_constant< int, 2 >{}, 16 );
         add( std::integral_constant< int, 32 >{}, std::integral_constant< int, 1 >{}, 50 );
         return products;
      } );
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
