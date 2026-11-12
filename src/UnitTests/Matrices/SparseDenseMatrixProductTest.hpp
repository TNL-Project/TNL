#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
#include <TNL/Containers/Array.h>
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

// Computes C = alpha * A * B + beta * C by sparseDenseMatrixProductWithLambdas, the lambda functions read B and update C.
// The lambda functions are defined in a named function, because nvcc does not allow extended lambdas inside lambdas.
template< TNL::Algorithms::Segments::ElementsOrganization Organization, typename Matrix, typename DenseMatrix, typename Real >
void
sparseDenseMatrixProductByLambdas( const Matrix& A, const DenseMatrix& B, DenseMatrix& C, Real alpha, Real beta )
{
   using IndexType = typename Matrix::IndexType;
   using ComputeRealType = typename Matrix::ComputeRealType;
   const auto BView = B.getConstView();
   const auto CSegments = C.getView().getSegments();
   auto CValues = C.getValues().getView();
   auto fetchB = [ = ] __cuda_callable__( IndexType column, IndexType j )
   {
      return BView( column, j );
   };
   auto storeC = [ = ] __cuda_callable__( IndexType row, int j, const ComputeRealType& value ) mutable
   {
      auto& element = CValues[ CSegments.getGlobalIndex( row, j ) ];
      element = alpha * value + beta * element;
   };
   TNL::Matrices::sparseDenseMatrixProductWithLambdas< Organization, testMaxTileSize >( A, B.getColumns(), fetchB, storeC );
}

// The same with B given by the dense matrix and only C by a lambda function.
template< typename Matrix, typename DenseMatrix, typename Real >
void
sparseDenseMatrixProductByStorer( const Matrix& A, const DenseMatrix& B, DenseMatrix& C, Real alpha, Real beta )
{
   using IndexType = typename Matrix::IndexType;
   using ComputeRealType = typename Matrix::ComputeRealType;
   const auto CSegments = C.getView().getSegments();
   auto CValues = C.getValues().getView();
   auto storeC = [ = ] __cuda_callable__( IndexType row, int j, const ComputeRealType& value ) mutable
   {
      auto& element = CValues[ CSegments.getGlobalIndex( row, j ) ];
      element = alpha * value + beta * element;
   };
   TNL::Matrices::sparseDenseMatrixProductWithLambdas< TNL::Algorithms::Segments::RowMajorOrder, testMaxTileSize >(
      A, B.getColumns(), B.getConstView(), storeC );
}

// Returns the stride of the array of the values of B for DirectFetch.
template< typename DenseMatrix >
typename DenseMatrix::IndexType
getDirectFetchStride( const DenseMatrix& B )
{
   const auto BSegments = B.getSegments().getConstView();
   if constexpr( DenseMatrix::getOrganization() == TNL::Algorithms::Segments::RowMajorOrder ) {
      if( B.getRows() > 1 )
         return BSegments.getGlobalIndex( 1, 0 ) - BSegments.getGlobalIndex( 0, 0 );
   }
   else if( B.getColumns() > 1 )
      return BSegments.getGlobalIndex( 0, 1 ) - BSegments.getGlobalIndex( 0, 0 );
   return 0;
}

// The same with B given by DirectFetch. With `transformed`, the values of B are doubled and stored in an array shifted
// by one element, so that it is not aligned for the vectorized loads, and the transformation halves them.
template< typename Matrix, typename DenseMatrix, typename Real >
void
sparseDenseMatrixProductByDirectFetch(
   const Matrix& A,
   const DenseMatrix& B,
   DenseMatrix& C,
   Real alpha,
   Real beta,
   bool transformed )
{
   using IndexType = typename Matrix::IndexType;
   using RealType = typename DenseMatrix::RealType;
   using ComputeRealType = typename Matrix::ComputeRealType;
   constexpr auto Organization = DenseMatrix::getOrganization();
   const IndexType stride = getDirectFetchStride( B );
   const auto CSegments = C.getView().getSegments();
   auto CValues = C.getValues().getView();
   auto storeC = [ = ] __cuda_callable__( IndexType row, int j, const ComputeRealType& value ) mutable
   {
      auto& element = CValues[ CSegments.getGlobalIndex( row, j ) ];
      element = alpha * value + beta * element;
   };
   if( ! transformed ) {
      TNL::Matrices::sparseDenseMatrixProductWithLambdas< Organization, testMaxTileSize >(
         A, B.getColumns(), TNL::Matrices::directFetch< Organization >( B.getValues().getData(), stride ), storeC );
      return;
   }
   const auto& values = B.getValues();
   typename DenseMatrix::ValuesVectorType shifted( values.getSize() + 1 );
   shifted.setElement( 0, 0 );
   auto shiftedView = shifted.getView();
   const auto valuesView = values.getConstView();
   TNL::Algorithms::parallelFor< typename DenseMatrix::DeviceType >(
      0,
      values.getSize(),
      [ = ] __cuda_callable__( IndexType i ) mutable
      {
         shiftedView[ i + 1 ] = RealType( 2 ) * valuesView[ i ];
      } );
   auto half = [] __cuda_callable__( const RealType& value, IndexType, IndexType ) -> RealType
   {
      return value / RealType( 2 );
   };
   TNL::Matrices::sparseDenseMatrixProductWithLambdas< Organization, testMaxTileSize >(
      A, B.getColumns(), TNL::Matrices::directFetch< Organization >( shifted.getData() + 1, stride, half ), storeC );
}

// The same with B given by DirectFetch reading an array of 16-bit integers (the values of B are small integers), which
// are narrower than the elements of B, so they are loaded by the vectorized loads of their bytes.
template< typename Matrix, typename DenseMatrix, typename Real >
void
sparseDenseMatrixProductByNarrowDirectFetch( const Matrix& A, const DenseMatrix& B, DenseMatrix& C, Real alpha, Real beta )
{
   using IndexType = typename Matrix::IndexType;
   using RealType = typename DenseMatrix::RealType;
   using ComputeRealType = typename Matrix::ComputeRealType;
   using DeviceType = typename DenseMatrix::DeviceType;
   constexpr auto Organization = DenseMatrix::getOrganization();
   const auto CSegments = C.getView().getSegments();
   auto CValues = C.getValues().getView();
   auto storeC = [ = ] __cuda_callable__( IndexType row, int j, const ComputeRealType& value ) mutable
   {
      auto& element = CValues[ CSegments.getGlobalIndex( row, j ) ];
      element = alpha * value + beta * element;
   };
   const auto valuesView = B.getValues().getConstView();
   TNL::Containers::Array< std::int16_t, DeviceType, IndexType > narrow( valuesView.getSize() );
   auto narrowView = narrow.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      valuesView.getSize(),
      [ = ] __cuda_callable__( IndexType i ) mutable
      {
         narrowView[ i ] = static_cast< std::int16_t >( valuesView[ i ] );
      } );
   auto widen = [] __cuda_callable__( const std::int16_t& value, IndexType, IndexType ) -> RealType
   {
      return static_cast< RealType >( value );
   };
   TNL::Matrices::sparseDenseMatrixProductWithLambdas< Organization, testMaxTileSize >(
      A,
      B.getColumns(),
      TNL::Matrices::directFetch< Organization >( narrow.getData(), getDirectFetchStride( B ), widen ),
      storeC );
}

// The same by sparseDenseMatrixProductWithLambdasLanesOverColumns, B is given by DirectFetch or by a lambda function.
template< int Lanes, int ColumnsPerLane, typename Matrix, typename DenseMatrix, typename Real >
void
sparseDenseMatrixProductByLambdasLanesOverColumns(
   const Matrix& A,
   const DenseMatrix& B,
   DenseMatrix& C,
   Real alpha,
   Real beta,
   bool direct )
{
   using IndexType = typename Matrix::IndexType;
   using ComputeRealType = typename Matrix::ComputeRealType;
   constexpr auto Organization = DenseMatrix::getOrganization();
   const auto BView = B.getConstView();
   const auto CSegments = C.getView().getSegments();
   auto CValues = C.getValues().getView();
   auto fetchB = [ = ] __cuda_callable__( IndexType column, IndexType j )
   {
      return BView( column, j );
   };
   auto storeC = [ = ] __cuda_callable__( IndexType row, int j, const ComputeRealType& value ) mutable
   {
      auto& element = CValues[ CSegments.getGlobalIndex( row, j ) ];
      element = alpha * value + beta * element;
   };
   if( direct )
      TNL::Matrices::sparseDenseMatrixProductWithLambdasLanesOverColumns< Lanes, ColumnsPerLane >(
         A,
         B.getColumns(),
         TNL::Matrices::directFetch< Organization >( B.getValues().getData(), getDirectFetchStride( B ) ),
         storeC );
   else
      TNL::Matrices::sparseDenseMatrixProductWithLambdasLanesOverColumns< Lanes, ColumnsPerLane, Organization >(
         A, B.getColumns(), fetchB, storeC );
}

// Tests sparseDenseMatrixProductWithLambdas with both organizations, i.e. also with the one not matching B.
template< typename Matrix, TNL::Algorithms::Segments::ElementsOrganization Organization >
void
test_SparseDenseMatrixProductWithLambdas()
{
   using RealType = typename Matrix::RealType;
   using Function = SparseDenseMatrixProductFunction< Matrix, Organization >;
   test_SparseDenseMatrixProductVariants< Matrix, Organization >(
      []( const Matrix& A )
      {
         std::vector< std::pair< std::string, Function > > products;
         products.emplace_back(
            "row-major access",
            [ &A ]( const auto& B, auto& C, RealType alpha, RealType beta )
            {
               sparseDenseMatrixProductByLambdas< TNL::Algorithms::Segments::RowMajorOrder >( A, B, C, alpha, beta );
            } );
         products.emplace_back(
            "column-major access",
            [ &A ]( const auto& B, auto& C, RealType alpha, RealType beta )
            {
               sparseDenseMatrixProductByLambdas< TNL::Algorithms::Segments::ColumnMajorOrder >( A, B, C, alpha, beta );
            } );
         products.emplace_back(
            "dense matrix B",
            [ &A ]( const auto& B, auto& C, RealType alpha, RealType beta )
            {
               sparseDenseMatrixProductByStorer( A, B, C, alpha, beta );
            } );
         products.emplace_back(
            "direct fetch",
            [ &A ]( const auto& B, auto& C, RealType alpha, RealType beta )
            {
               sparseDenseMatrixProductByDirectFetch( A, B, C, alpha, beta, false );
            } );
         products.emplace_back(
            "direct fetch, unaligned and transformed",
            [ &A ]( const auto& B, auto& C, RealType alpha, RealType beta )
            {
               sparseDenseMatrixProductByDirectFetch( A, B, C, alpha, beta, true );
            } );
         if constexpr( std::is_arithmetic_v< RealType > )
            products.emplace_back(
               "direct fetch of a narrower type",
               [ &A ]( const auto& B, auto& C, RealType alpha, RealType beta )
               {
                  sparseDenseMatrixProductByNarrowDirectFetch( A, B, C, alpha, beta );
               } );
         products.emplace_back(
            "lanes over columns, Lanes = 4, columns per lane = 2, direct fetch",
            [ &A ]( const auto& B, auto& C, RealType alpha, RealType beta )
            {
               sparseDenseMatrixProductByLambdasLanesOverColumns< 4, 2 >( A, B, C, alpha, beta, true );
            } );
         products.emplace_back(
            "lanes over columns, Lanes = 32, columns per lane = 1, lambda function",
            [ &A ]( const auto& B, auto& C, RealType alpha, RealType beta )
            {
               sparseDenseMatrixProductByLambdasLanesOverColumns< 32, 1 >( A, B, C, alpha, beta, false );
            } );
         return products;
      } );
}

// Tests C = alpha * B * op( A ) + beta * C computed by denseSparseMatrixProduct for both op( A ) = A and op( A ) = A^T
// with the same matrix A as test_SparseDenseMatrixProductVariants. All values are small integers, so the results are
// exact for all tested types.
template< typename Matrix, TNL::Algorithms::Segments::ElementsOrganization Organization >
void
test_DenseSparseMatrixProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using DenseMatrixType = TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType, Organization >;
   using HostDenseMatrixType = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Host, IndexType, Organization >;

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

   for( const auto transposeA : { TNL::Matrices::TransposeState::None, TNL::Matrices::TransposeState::Transpose } ) {
      const bool transposed = transposeA == TNL::Matrices::TransposeState::Transpose;
      // op( A ) has n rows and m columns
      const IndexType n = transposed ? columns : rows;
      const IndexType m = transposed ? rows : columns;
      for( const IndexType k : { 1, 3, 5, 9, 17, 33, 70 } ) {
         HostDenseMatrixType hostB( k, n );
         HostDenseMatrixType hostC0( k, m );
         for( IndexType i = 0; i < k; i++ )
            for( IndexType j = 0; j < n; j++ )
               hostB.setElement( i, j, RealType( j % 7 + i + 1 ) );
         for( IndexType i = 0; i < k; i++ )
            for( IndexType j = 0; j < m; j++ )
               hostC0.setElement( i, j, RealType( j % 11 - i ) );
         DenseMatrixType B;
         B = hostB;

         for( const auto& [ alpha, beta ] : { std::pair< int, int >{ 1, 0 }, std::pair< int, int >{ 2, -1 } } ) {
            SCOPED_TRACE(
               std::string( transposed ? "transposed A" : "A" ) + ", k = " + std::to_string( k )
               + ", alpha = " + std::to_string( alpha ) + ", beta = " + std::to_string( beta ) );
            HostDenseMatrixType expected( hostC0 );
            for( IndexType i = 0; i < k; i++ )
               for( IndexType j = 0; j < m; j++ )
                  expected.setElement( i, j, RealType( beta ) * hostC0.getElement( i, j ) );
            for( const auto& [ position, value ] : elements ) {
               // the element of op( A ) in the row l and the column j
               const IndexType l = transposed ? position.second : position.first;
               const IndexType j = transposed ? position.first : position.second;
               for( IndexType i = 0; i < k; i++ )
                  expected.addElement( i, j, RealType( alpha ) * hostB.getElement( i, l ) * value, RealType( 1 ) );
            }

            DenseMatrixType C;
            C = hostC0;
            TNL::Matrices::denseSparseMatrixProduct< testMaxTileSize >(
               B, A, C, RealType( alpha ), RealType( beta ), transposeA );
            HostDenseMatrixType hostC;
            hostC = C;
            for( IndexType i = 0; i < k; i++ )
               for( IndexType j = 0; j < m; j++ )
                  EXPECT_EQ( hostC.getElement( i, j ), expected.getElement( i, j ) ) << "i = " << i << ", j = " << j;
         }
      }
   }

   DenseMatrixType C( 3, columns );
   EXPECT_THROW( TNL::Matrices::denseSparseMatrixProduct( DenseMatrixType( 3, rows + 1 ), A, C ), std::invalid_argument );
   EXPECT_THROW( TNL::Matrices::denseSparseMatrixProduct( DenseMatrixType( 4, rows ), A, C ), std::invalid_argument );
   EXPECT_THROW(
      TNL::Matrices::denseSparseMatrixProduct(
         DenseMatrixType( 3, rows ), A, C, RealType( 1 ), RealType( 0 ), TNL::Matrices::TransposeState::Transpose ),
      std::invalid_argument );
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
