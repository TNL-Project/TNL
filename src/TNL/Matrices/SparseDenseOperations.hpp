// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/multiReduce.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Functional.h>
#include <TNL/Math.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/TypeTraits.h>
#include "SparseDenseOperations.h"

namespace TNL::Matrices {

namespace detail {

// Native vector types for the vectorized loads of N consecutive values.
template< typename Real, int N >
struct VectorType
{
   static constexpr bool exists = false;
};

#if defined( __CUDACC__ ) || defined( __HIP__ )
template<>
struct VectorType< float, 2 >
{
   static constexpr bool exists = true;
   using type = float2;
};

template<>
struct VectorType< float, 4 >
{
   static constexpr bool exists = true;
   using type = float4;
};

template<>
struct VectorType< double, 2 >
{
   static constexpr bool exists = true;
   using type = double2;
};
#endif

// Width of the widest native vector type which divides N.
template< typename Real, int N >
constexpr int
vectorWidth()
{
   if constexpr( N % 4 == 0 && VectorType< Real, 4 >::exists )
      return 4;
   else if constexpr( N % 2 == 0 && VectorType< Real, 2 >::exists )
      return 2;
   else
      return 1;
}

// Loads N consecutive values. When `aligned` is true, the source must be aligned to the width of the vector type
// given by vectorWidth< Real, N >() and the values are loaded by the vectorized loads in device code.
template< int N, typename Real >
__cuda_callable__
void
loadConsecutive( const Real* source, Real ( &target )[ N ], bool aligned )
{
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   constexpr int W = vectorWidth< Real, N >();
   if constexpr( W > 1 ) {
      if( aligned ) {
         using Vector = typename VectorType< Real, W >::type;
         const Vector* vectorSource = reinterpret_cast< const Vector* >( source );
         for( int i = 0; i < N / W; i++ ) {
            const Vector v = vectorSource[ i ];
            const Real* scalars = reinterpret_cast< const Real* >( &v );
            for( int j = 0; j < W; j++ )
               target[ i * W + j ] = scalars[ j ];
         }
         return;
      }
   }
#endif
   for( int i = 0; i < N; i++ )
      target[ i ] = source[ i ];
}

// The fetch of the multireduction computing the columns columnOffset + j of A * B.
// It is a function object, because nvcc does not allow returning an extended lambda from a function with
// a deduced return type.
template< typename SparseMatrixView, typename DenseMatrixBView >
struct SparseDenseMatrixProductFetch
{
   using IndexType = typename SparseMatrixView::IndexType;
   using ComputeRealType = typename SparseMatrixView::ComputeRealType;

   SparseDenseMatrixProductFetch( const SparseMatrixView& A, const DenseMatrixBView& B, IndexType columnOffset )
   : values( A.getValues().getConstView() ),
     columnIndexes( A.getColumnIndexes().getConstView() ),
     B( B ),
     columnOffset( columnOffset )
   {}

   __cuda_callable__
   ComputeRealType
   operator()( IndexType globalIdx, int j ) const
   {
      const IndexType column = columnIndexes[ globalIdx ];
      TNL_ASSERT_TRUE( column >= 0 || column == paddingIndex< IndexType >, "Wrong column index." );
      // The check of the padding elements is a branch, which separates the multiplication from the
      // addition in the reduction, so the compiler cannot fuse them into an FMA instruction. This matters
      // when the product is compute-bound (e.g. in double precision), hence the check is done only for
      // segments with padding elements.
      if constexpr( SparseMatrixView::SegmentsViewType::havePadding() ) {
         if( column == paddingIndex< IndexType > )
            return 0;
      }
      if constexpr( SparseMatrixView::isBinary() )
         return B( column, columnOffset + j );
      else
         return values[ globalIdx ] * B( column, columnOffset + j );
   }

   // Computes the columns columnOffset + firstJ, ..., columnOffset + firstJ + Count - 1 at once, the elements of B
   // are read by vectorized loads when vectorLoads is set. It is a template, so that callableArgumentCount still
   // detects the fetch with two arguments (the template parameter cannot be deduced from a generic argument).
   template< int Count >
   __cuda_callable__
   Containers::StaticArray< Count, ComputeRealType >
   operator()( IndexType globalIdx, int firstJ, std::integral_constant< int, Count > ) const
   {
      const IndexType column = columnIndexes[ globalIdx ];
      TNL_ASSERT_TRUE( column >= 0 || column == paddingIndex< IndexType >, "Wrong column index." );
      Containers::StaticArray< Count, ComputeRealType > result;
      // see the check of the padding elements above
      if constexpr( SparseMatrixView::SegmentsViewType::havePadding() ) {
         if( column == paddingIndex< IndexType > ) {
            result = ComputeRealType{ 0 };
            return result;
         }
      }
      BRealType b[ Count ];
      // the consecutive columns of B are consecutive in memory only in the row-major order
      if constexpr( DenseMatrixBView::getOrganization() == Algorithms::Segments::RowMajorOrder )
         loadConsecutive< Count >( &B( column, columnOffset + firstJ ), b, vectorLoads );
      else
         for( int c = 0; c < Count; c++ )
            b[ c ] = B( column, columnOffset + firstJ + c );
      for( int c = 0; c < Count; c++ ) {
         if constexpr( SparseMatrixView::isBinary() )
            result[ c ] = b[ c ];
         else
            result[ c ] = values[ globalIdx ] * b[ c ];
      }
      return result;
   }

   using BRealType = std::remove_const_t< typename DenseMatrixBView::RealType >;

   std::decay_t< decltype( std::declval< SparseMatrixView >().getValues().getConstView() ) > values;
   std::decay_t< decltype( std::declval< SparseMatrixView >().getColumnIndexes().getConstView() ) > columnIndexes;
   DenseMatrixBView B;
   IndexType columnOffset;
   // the rows of B are aligned for the vectorized loads of the template operator()
   bool vectorLoads = false;
};

// The storer of the multireduction computing the columns columnOffset + j of C = alpha * A * B + beta * C.
template< typename ComputeRealType, typename DenseMatrixCView, typename Real >
struct SparseDenseMatrixProductStorer
{
   using IndexType = typename DenseMatrixCView::IndexType;

   __cuda_callable__
   void
   operator()( IndexType row, int j, const ComputeRealType& value ) const
   {
      auto& element = C( row, columnOffset + j );
      if( beta == Real{ 0 } )
         element = alpha * value;
      else
         element = alpha * value + beta * element;
   }

   // mutable, because the operator is const and the view gives a reference to its element only when not const
   mutable DenseMatrixCView C;
   IndexType columnOffset;
   Real alpha;
   Real beta;
};

// Checks the dimensions of the matrices of the product C = alpha * A * B + beta * C.
template< typename SparseMatrix, typename DenseMatrixB, typename DenseMatrixC >
void
checkSparseDenseMatrixProductDimensions( const SparseMatrix& A, const DenseMatrixB& B, const DenseMatrixC& C )
{
   if( A.getColumns() != B.getRows() )
      throw std::invalid_argument(
         "sparseDenseMatrixProduct: the number of columns of A does not match the number of rows of B" );
   if( A.getRows() != C.getRows() )
      throw std::invalid_argument( "sparseDenseMatrixProduct: the number of rows of A does not match the number of rows of C" );
   if( B.getColumns() != C.getColumns() )
      throw std::invalid_argument(
         "sparseDenseMatrixProduct: the number of columns of B does not match the number of columns of C" );
}

}  // namespace detail

template< int MaxTileSize, typename SparseMatrix, typename DenseMatrixB, typename DenseMatrixC, typename Real >
void
sparseDenseMatrixProduct(
   const SparseMatrix& A,
   const DenseMatrixB& B,
   DenseMatrixC& C,
   Real alpha,
   Real beta,
   const Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   static_assert( MaxTileSize > 0, "The maximal tile size must be positive." );
   static_assert( ! SparseMatrix::isSymmetric(), "sparseDenseMatrixProduct does not support symmetric matrices yet." );
   using IndexType = typename SparseMatrix::IndexType;
   detail::checkSparseDenseMatrixProductDimensions( A, B, C );

   const IndexType k = B.getColumns();
   // For more than 4 columns of row-major B, the default configuration of CSR on GPUs distributes the threads of the
   // rows over the columns, the tile of the columns processed at once is the smallest power of two not smaller than k
   // (from 8 to 128). In single precision, each thread computes up to 4 columns, so that they are loaded by
   // vectorized loads of 16 bytes, and there are at least 4 threads per row. In double precision, a row is processed
   // by up to a warp of threads. This was the best choice in the benchmarks on RTX 3060 and A100.
   using SegmentsType = std::decay_t< decltype( A.getSegments() ) >;
   using Algorithms::Segments::ThreadsToSegmentsMapping;
   Algorithms::Segments::LaunchConfiguration config = launchConfig;
   if constexpr( Algorithms::Segments::isCSRSegments_v< SegmentsType >
                 && std::is_same_v< typename SparseMatrix::DeviceType, Devices::GPU > )
   {
      if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Default ) {
         if constexpr( DenseMatrixB::getOrganization() == Algorithms::Segments::RowMajorOrder ) {
            if( k > 4 ) {
               auto multiply = [ & ]( auto tileSize )
               {
                  constexpr int TileSize = decltype( tileSize )::value;
                  constexpr bool singlePrecision = sizeof( std::remove_const_t< typename DenseMatrixB::RealType > ) <= 4;
                  constexpr int Lanes = singlePrecision ? TileSize / TNL::min( 4, TileSize / 4 ) : TNL::min( 32, TileSize );
                  sparseDenseMatrixProductLanesOverColumns< Lanes, TileSize / Lanes >( A, B, C, alpha, beta );
               };
               if( k <= 8 )
                  multiply( std::integral_constant< int, 8 >{} );
               else if( k <= 16 )
                  multiply( std::integral_constant< int, 16 >{} );
               else if( k <= 32 )
                  multiply( std::integral_constant< int, 32 >{} );
               else if( k <= 64 )
                  multiply( std::integral_constant< int, 64 >{} );
               else
                  multiply( std::integral_constant< int, 128 >{} );
               return;
            }
         }
         // Otherwise, the multireduction with the threads of the rows distributed over their elements is used. The rows
         // with a few elements on average are processed by the Tiered mapping, which adapts to the irregular lengths of
         // the rows. The longer rows are processed by 16 threads per row, the warp size of the default configuration of
         // CSR would leave most threads idle.
         const IndexType averageRowLength = A.getRows() > 0 ? A.getColumnIndexes().getSize() / A.getRows() : 0;
         if( averageRowLength < 16 ) {
            config.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::Tiered );
            config.setThreadsPerSegmentCount( k <= 8 ? 4 : 2 );
         }
         else {
            config.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::Uniform );
            config.setThreadsPerSegmentCount( 16 );
         }
      }
   }

   if( k == 0 )
      return;
   if( k > std::numeric_limits< int >::max() )
      throw std::invalid_argument( "sparseDenseMatrixProduct: too many columns of B" );
   using ComputeRealType = typename SparseMatrix::ComputeRealType;
   const auto AView = A.getConstView();
   const auto BView = B.getConstView();
   auto CView = C.getView();
   Algorithms::Segments::multiReduceAllSegmentsInTiles< MaxTileSize >(
      AView.getSegments(),
      static_cast< int >( k ),
      detail::SparseDenseMatrixProductFetch< decltype( AView ), decltype( BView ) >( AView, BView, 0 ),
      TNL::Plus{},
      detail::SparseDenseMatrixProductStorer< ComputeRealType, decltype( CView ), Real >{ CView, 0, alpha, beta },
      ComputeRealType{ 0 },
      config );
}

template< int Lanes, int ColumnsPerLane, typename SparseMatrix, typename DenseMatrixB, typename DenseMatrixC, typename Real >
void
sparseDenseMatrixProductLanesOverColumns(
   const SparseMatrix& A,
   const DenseMatrixB& B,
   DenseMatrixC& C,
   Real alpha,
   Real beta,
   typename SparseMatrix::IndexType maxRowLength )
{
   static_assert( ! SparseMatrix::isSymmetric(), "sparseDenseMatrixProduct does not support symmetric matrices yet." );
   using IndexType = typename SparseMatrix::IndexType;
   using ComputeRealType = typename SparseMatrix::ComputeRealType;
   detail::checkSparseDenseMatrixProductDimensions( A, B, C );

   const auto AView = A.getConstView();
   const auto BView = B.getConstView();
   auto CView = C.getView();
   const IndexType k = B.getColumns();
   if( k == 0 )
      return;

   // The vectorized loads of the consecutive columns of B computed by one thread need row-major B with the rows
   // aligned to the vector width. The tiles and the first columns of the threads start at multiples of the width.
   using BRealType = std::remove_const_t< typename DenseMatrixB::RealType >;
   constexpr int W = detail::vectorWidth< BRealType, ColumnsPerLane >();
   bool vectorLoads = false;
   if constexpr( W > 1 && DenseMatrixB::getOrganization() == Algorithms::Segments::RowMajorOrder ) {
      const auto& segments = BView.getSegments();
      const auto address = reinterpret_cast< std::uintptr_t >( BView.getValues().getData() );
      vectorLoads = address % ( W * sizeof( BRealType ) ) == 0
                 && ( BView.getRows() < 2 || ( segments.getGlobalIndex( 1, 0 ) - segments.getGlobalIndex( 0, 0 ) ) % W == 0 );
   }

   // the default is computed here from the number of elements of A known on the host, so that the multireduction
   // does not need to read it from the device
   using SegmentsViewType = typename std::decay_t< decltype( AView.getSegments() ) >::ConstViewType;
   if constexpr( Algorithms::Segments::isCSRSegments_v< SegmentsViewType >
                 && std::is_same_v< typename SparseMatrix::DeviceType, Devices::GPU > )
      if( maxRowLength <= 0 )
         maxRowLength = Algorithms::Segments::detail::MultiReducingOperations<
            SegmentsViewType >::template getDefaultMaxSegmentSize< Lanes >( A.getColumnIndexes().getSize() );

   // all columns are processed at once, the multireduction splits them into tiles processed in parallel
   if( k > std::numeric_limits< int >::max() )
      throw std::invalid_argument( "sparseDenseMatrixProductLanesOverColumns: too many columns of B" );
   detail::SparseDenseMatrixProductFetch< decltype( AView ), decltype( BView ) > fetch( AView, BView, 0 );
   fetch.vectorLoads = vectorLoads;
   Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions< Lanes, ColumnsPerLane >(
      AView.getSegments(),
      static_cast< int >( k ),
      fetch,
      TNL::Plus{},
      detail::SparseDenseMatrixProductStorer< ComputeRealType, decltype( CView ), Real >{ CView, 0, alpha, beta },
      ComputeRealType{ 0 },
      maxRowLength );
}

}  // namespace TNL::Matrices
