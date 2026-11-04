// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
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

// Computes the columns [columnOffset, columnOffset + TileSize) of C = alpha * A * B + beta * C.
template< int TileSize, typename SparseMatrixView, typename DenseMatrixBView, typename DenseMatrixCView, typename Real >
void
sparseDenseMatrixProductTile(
   const SparseMatrixView& A,
   const DenseMatrixBView& B,
   DenseMatrixCView& C,
   typename SparseMatrixView::IndexType columnOffset,
   Real alpha,
   Real beta,
   const Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   using ComputeRealType = typename SparseMatrixView::ComputeRealType;
   Algorithms::Segments::multiReduceAllSegments< TileSize >(
      A.getSegments(),
      SparseDenseMatrixProductFetch< SparseMatrixView, DenseMatrixBView >( A, B, columnOffset ),
      TNL::Plus{},
      SparseDenseMatrixProductStorer< ComputeRealType, DenseMatrixCView, Real >{ C, columnOffset, alpha, beta },
      ComputeRealType{ 0 },
      launchConfig );
}

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

// Calls f( std::integral_constant< int, size >{} ) for size in [1, MaxSize].
template< int MaxSize, typename Function >
void
dispatchTileSize( int size, Function&& f )
{
   if constexpr( MaxSize > 0 ) {
      if( size == MaxSize )
         f( std::integral_constant< int, MaxSize >{} );
      else
         dispatchTileSize< MaxSize - 1 >( size, std::forward< Function >( f ) );
   }
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
   // For more than 8 columns of row-major B, the default configuration of CSR on GPUs distributes the threads of the
   // rows over the columns, the tile of the columns processed at once is the smallest power of two not smaller than k
   // (at most 128). Each thread computes 4 columns, so that they can be loaded by vectorized loads, except for the
   // smallest tiles, which would have too few threads per row.
   using SegmentsType = std::decay_t< decltype( A.getSegments() ) >;
   if constexpr( Algorithms::Segments::isCSRSegments_v< SegmentsType >
                 && std::is_same_v< typename SparseMatrix::DeviceType, Devices::GPU >
                 && DenseMatrixB::getOrganization() == Algorithms::Segments::RowMajorOrder )
   {
      if( launchConfig.getThreadsToSegmentsMapping() == Algorithms::Segments::ThreadsToSegmentsMapping::Default && k > 8 ) {
         if( k <= 16 )
            sparseDenseMatrixProductLanesOverColumns< 8, 2 >( A, B, C, alpha, beta );
         else if( k <= 32 )
            sparseDenseMatrixProductLanesOverColumns< 8, 4 >( A, B, C, alpha, beta );
         else if( k <= 64 )
            sparseDenseMatrixProductLanesOverColumns< 16, 4 >( A, B, C, alpha, beta );
         else
            sparseDenseMatrixProductLanesOverColumns< 32, 4 >( A, B, C, alpha, beta );
         return;
      }
   }

   const auto AView = A.getConstView();
   const auto BView = B.getConstView();
   auto CView = C.getView();
   for( IndexType columnOffset = 0; columnOffset < k; columnOffset += MaxTileSize ) {
      const int tileSize = TNL::min( k - columnOffset, static_cast< IndexType >( MaxTileSize ) );
      detail::dispatchTileSize< MaxTileSize >(
         tileSize,
         [ & ]( auto tileSize_ )
         {
            detail::sparseDenseMatrixProductTile< decltype( tileSize_ )::value >(
               AView, BView, CView, columnOffset, alpha, beta, launchConfig );
         } );
   }
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
   constexpr int TileSize = Lanes * ColumnsPerLane;

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
   // does not need to read it from the device for every tile
   using SegmentsViewType = typename std::decay_t< decltype( AView.getSegments() ) >::ConstViewType;
   if constexpr( Algorithms::Segments::isCSRSegments_v< SegmentsViewType >
                 && std::is_same_v< typename SparseMatrix::DeviceType, Devices::GPU > )
      if( maxRowLength <= 0 )
         maxRowLength = Algorithms::Segments::detail::MultiReducingOperations<
            SegmentsViewType >::template getDefaultMaxSegmentSize< Lanes >( A.getColumnIndexes().getSize() );

   for( IndexType columnOffset = 0; columnOffset < k; columnOffset += TileSize ) {
      detail::SparseDenseMatrixProductFetch< decltype( AView ), decltype( BView ) > fetch( AView, BView, columnOffset );
      fetch.vectorLoads = vectorLoads;
      Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions< Lanes, ColumnsPerLane >(
         AView.getSegments(),
         TNL::min( k - columnOffset, static_cast< IndexType >( TileSize ) ),
         fetch,
         TNL::Plus{},
         detail::SparseDenseMatrixProductStorer< ComputeRealType, decltype( CView ), Real >{ CView, columnOffset, alpha, beta },
         ComputeRealType{ 0 },
         maxRowLength );
   }
}

}  // namespace TNL::Matrices
