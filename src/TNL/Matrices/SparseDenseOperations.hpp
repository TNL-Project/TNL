// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/multiReduce.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Functional.h>
#include <TNL/Math.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/TypeTraits.h>
#include <TNL/SystemInfo.h>
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

// Native vector types of Bytes bytes for the vectorized loads of the values narrower than the elements of B.
template< int Bytes >
struct RawVectorType
{
   static constexpr bool exists = false;
};

#if defined( __CUDACC__ ) || defined( __HIP__ )
template<>
struct RawVectorType< 4 >
{
   static constexpr bool exists = true;
   using type = unsigned int;
};

template<>
struct RawVectorType< 8 >
{
   static constexpr bool exists = true;
   using type = uint2;
};

template<>
struct RawVectorType< 16 >
{
   static constexpr bool exists = true;
   using type = uint4;
};
#endif

// Loads N consecutive values by the vectors of W values. When `aligned` is true, the source must be aligned to the
// size of the vector.
template< int N, int W, typename Value >
__cuda_callable__
void
loadConsecutiveRaw( const Value* source, Value ( &target )[ N ], bool aligned )
{
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   using Raw = RawVectorType< W * sizeof( Value ) >;
   if constexpr( W > 1 && N % W == 0 && Raw::exists ) {
      if( aligned ) {
         const auto* vectorSource = reinterpret_cast< const typename Raw::type* >( source );
         for( int i = 0; i < N / W; i++ ) {
            const typename Raw::type v = vectorSource[ i ];
            const Value* scalars = reinterpret_cast< const Value* >( &v );
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

// The access to the elements of a dense matrix B for the fetch of the product. The consecutive columns of row-major B
// are loaded by vectorized loads.
template< typename DenseMatrixBView >
struct DenseMatrixBAccess
{
   using IndexType = typename DenseMatrixBView::IndexType;
   using RealType = std::remove_const_t< typename DenseMatrixBView::RealType >;

   __cuda_callable__
   RealType
   operator()( IndexType column, IndexType j ) const
   {
      return B( column, j );
   }

   // Loads the columns firstJ, ..., firstJ + Count - 1 of the row `column`, by the vectorized loads when vectorLoads
   // is set.
   template< int Count >
   __cuda_callable__
   void
   load( IndexType column, IndexType firstJ, RealType ( &b )[ Count ], bool vectorLoads ) const
   {
      // the consecutive columns of B are consecutive in memory only in the row-major order
      if constexpr( DenseMatrixBView::getOrganization() == Algorithms::Segments::RowMajorOrder )
         loadConsecutive< Count >( &B( column, firstJ ), b, vectorLoads );
      else
         for( int c = 0; c < Count; c++ )
            b[ c ] = B( column, firstJ + c );
   }

   // Tells whether W consecutive columns starting at multiples of W can be loaded by the vectorized loads, i.e. whether
   // B is row-major with the rows aligned to the vector width.
   template< int W >
   [[nodiscard]] bool
   canLoadVectors() const
   {
      if constexpr( DenseMatrixBView::getOrganization() == Algorithms::Segments::RowMajorOrder ) {
         const auto& segments = B.getSegments();
         const auto address = reinterpret_cast< std::uintptr_t >( B.getValues().getData() );
         return address % ( W * sizeof( RealType ) ) == 0
             && ( B.getRows() < 2 || ( segments.getGlobalIndex( 1, 0 ) - segments.getGlobalIndex( 0, 0 ) ) % W == 0 );
      }
      return false;
   }

   DenseMatrixBView B;
};

// The access to the elements of B given by the function fetchB( column, j ).
template< typename FetchB, typename Index >
struct LambdaBAccess
{
   using IndexType = Index;
   using RealType = std::decay_t< decltype( std::declval< const FetchB& >()( Index{}, Index{} ) ) >;

   __cuda_callable__
   RealType
   operator()( IndexType column, IndexType j ) const
   {
      return fetchB( column, j );
   }

   template< int Count >
   __cuda_callable__
   void
   load( IndexType column, IndexType firstJ, RealType ( &b )[ Count ], bool ) const
   {
      for( int c = 0; c < Count; c++ )
         b[ c ] = fetchB( column, firstJ + c );
   }

   template< int W >
   [[nodiscard]] bool
   canLoadVectors() const
   {
      return false;
   }

   FetchB fetchB;
};

// The access to the elements of B given by DirectFetch: the elements are loaded from the array like from a dense
// matrix (by vectorized loads for row-major B) and then transformed.
template< typename DirectFetch >
struct DirectBAccess
{
   using IndexType = std::decay_t< decltype( std::declval< DirectFetch >().stride ) >;
   using ValueType = std::remove_const_t< std::remove_pointer_t< decltype( std::declval< DirectFetch >().data ) > >;
   using RealType = std::decay_t< decltype( std::declval< const DirectFetch& >()( IndexType{}, IndexType{} ) ) >;

   __cuda_callable__
   RealType
   operator()( IndexType column, IndexType j ) const
   {
      return fetch( column, j );
   }

   template< int Count >
   __cuda_callable__
   void
   load( IndexType column, IndexType firstJ, RealType ( &b )[ Count ], bool vectorLoads ) const
   {
      if constexpr( DirectFetch::getOrganization() == Algorithms::Segments::RowMajorOrder ) {
         ValueType values[ Count ];
         if constexpr( std::is_same_v< ValueType, RealType > )
            loadConsecutive< Count >( fetch.data + fetch.getIndex( column, firstJ ), values, vectorLoads );
         else
            loadConsecutiveRaw< Count, vectorWidth< RealType, Count >() >(
               fetch.data + fetch.getIndex( column, firstJ ), values, vectorLoads );
         for( int c = 0; c < Count; c++ )
            b[ c ] = fetch.transform( values[ c ], column, firstJ + c );
      }
      else
         for( int c = 0; c < Count; c++ )
            b[ c ] = fetch( column, firstJ + c );
   }

   // The width W of the vectors, to which the columns of the threads are aligned, is given by the type of the elements
   // of B. The values of other types (e.g. half precision) are loaded by the vectors of W values of the same size.
   template< int W >
   [[nodiscard]] bool
   canLoadVectors() const
   {
      if constexpr( DirectFetch::getOrganization() == Algorithms::Segments::RowMajorOrder
                    && ( std::is_same_v< ValueType, RealType > || RawVectorType< W * sizeof( ValueType ) >::exists ) )
      {
         const auto address = reinterpret_cast< std::uintptr_t >( fetch.data );
         return address % ( W * sizeof( ValueType ) ) == 0 && fetch.stride % W == 0;
      }
      return false;
   }

   DirectFetch fetch;
};

// The fetch of the multireduction computing the columns columnOffset + j of A * B, the elements of B are read by
// BAccess (DenseMatrixBAccess or LambdaBAccess).
// It is a function object, because nvcc does not allow returning an extended lambda from a function with
// a deduced return type.
template< typename SparseMatrixView, typename BAccess >
struct SparseDenseMatrixProductFetch
{
   using IndexType = typename SparseMatrixView::IndexType;
   using ComputeRealType = typename SparseMatrixView::ComputeRealType;

   SparseDenseMatrixProductFetch( const SparseMatrixView& A, const BAccess& B, IndexType columnOffset )
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
      B.template load< Count >( column, columnOffset + firstJ, b, vectorLoads );
      for( int c = 0; c < Count; c++ ) {
         if constexpr( SparseMatrixView::isBinary() )
            result[ c ] = b[ c ];
         else
            result[ c ] = values[ globalIdx ] * b[ c ];
      }
      return result;
   }

   using BRealType = typename BAccess::RealType;

   std::decay_t< decltype( std::declval< SparseMatrixView >().getValues().getConstView() ) > values;
   std::decay_t< decltype( std::declval< SparseMatrixView >().getColumnIndexes().getConstView() ) > columnIndexes;
   BAccess B;
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

namespace detail {

// sparseDenseMatrixProductLanesOverColumns with the elements of B read by BAccess and the results passed to storer.
template< int Lanes, int ColumnsPerLane, typename SparseMatrix, typename BAccess, typename Storer >
void
sparseDenseMatrixProductLanesOverColumns(
   const SparseMatrix& A,
   typename SparseMatrix::IndexType k,
   const BAccess& B,
   const Storer& storer,
   typename SparseMatrix::IndexType maxRowLength )
{
   static_assert( ! SparseMatrix::isSymmetric(), "sparseDenseMatrixProduct does not support symmetric matrices yet." );
   using ComputeRealType = typename SparseMatrix::ComputeRealType;
   if( k == 0 )
      return;

   // The vectorized loads of the consecutive columns of B computed by one thread need row-major B with the rows
   // aligned to the vector width. The tiles and the first columns of the threads start at multiples of the width.
   constexpr int W = vectorWidth< typename BAccess::RealType, ColumnsPerLane >();
   bool vectorLoads = false;
   if constexpr( W > 1 )
      vectorLoads = B.template canLoadVectors< W >();

   // the default is computed here from the number of elements of A known on the host, so that the multireduction
   // does not need to read it from the device
   const auto AView = A.getConstView();
   using SegmentsViewType = typename std::decay_t< decltype( AView.getSegments() ) >::ConstViewType;
   if constexpr( Algorithms::Segments::isCSRSegments_v< SegmentsViewType >
                 && std::is_same_v< typename SparseMatrix::DeviceType, Devices::GPU > )
      if( maxRowLength <= 0 )
         maxRowLength = Algorithms::Segments::detail::MultiReducingOperations<
            SegmentsViewType >::template getDefaultMaxSegmentSize< Lanes >( A.getColumnIndexes().getSize() );

   // all columns are processed at once, the multireduction splits them into tiles processed in parallel
   if( k > std::numeric_limits< int >::max() )
      throw std::invalid_argument( "sparseDenseMatrixProductLanesOverColumns: too many columns of B" );
   SparseDenseMatrixProductFetch< decltype( AView ), BAccess > fetch( AView, B, 0 );
   fetch.vectorLoads = vectorLoads;
   Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions< Lanes, ColumnsPerLane >(
      AView.getSegments(), static_cast< int >( k ), fetch, TNL::Plus{}, storer, ComputeRealType{ 0 }, maxRowLength );
}

// sparseDenseMatrixProduct with the elements of B read by BAccess and the results passed to storer. Organization
// tells how B is stored, i.e. whether consecutive columns of B are consecutive in memory.
template<
   int MaxTileSize,
   Algorithms::Segments::ElementsOrganization Organization,
   typename SparseMatrix,
   typename BAccess,
   typename Storer >
void
sparseDenseMatrixProduct(
   const SparseMatrix& A,
   typename SparseMatrix::IndexType k,
   const BAccess& B,
   const Storer& storer,
   const Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   static_assert( MaxTileSize > 0, "The maximal tile size must be positive." );
   static_assert( ! SparseMatrix::isSymmetric(), "sparseDenseMatrixProduct does not support symmetric matrices yet." );
   using IndexType = typename SparseMatrix::IndexType;
   using BRealType = typename BAccess::RealType;

   // For more than 4 columns of row-major B, the default configuration of CSR on GPUs distributes the threads of the
   // rows over the columns, the tile of the columns processed at once is the smallest power of two not smaller than k
   // (from 8 to 128). In single precision, each thread computes up to 4 columns, so that they are loaded by
   // vectorized loads of 16 bytes, and there are at least 4 threads per row. In double precision, a row is processed
   // by up to a warp of threads. This was the best choice in the benchmarks on RTX 3060 and A100.
   using SegmentsType = std::decay_t< decltype( A.getSegments() ) >;
   using Algorithms::Segments::ThreadsToSegmentsMapping;
   Algorithms::Segments::LaunchConfiguration config = launchConfig;
   using DeviceType = typename SparseMatrix::DeviceType;
   // on the host, sparseDenseMatrixProductLanesOverColumns computes all columns of a row of C at once, which suits
   // row-major B, and the launch configuration is not used
   if constexpr(
      Algorithms::Segments::isCSRSegments_v< SegmentsType >
      && (std::is_same_v< DeviceType, Devices::GPU > || std::is_same_v< DeviceType, Devices::Host > || std::is_same_v< DeviceType, Devices::Sequential >) )
   {
      if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Default ) {
         if constexpr( Organization == Algorithms::Segments::RowMajorOrder ) {
            // on the host, all columns of a row are computed at once for any k, with the tile of the smallest power of
            // two not smaller than k, so that the inner loop over the columns of the tile has a constant trip count
            constexpr bool host = ! std::is_same_v< DeviceType, Devices::GPU >;
            if( k > 4 || host ) {
               auto multiply = [ & ]( auto tileSize )
               {
                  constexpr int TileSize = decltype( tileSize )::value;
                  constexpr bool singlePrecision = sizeof( BRealType ) <= 4;
                  constexpr int Lanes =
                     singlePrecision ? TileSize / TNL::max( 1, TNL::min( 4, TileSize / 4 ) ) : TNL::min( 32, TileSize );
                  sparseDenseMatrixProductLanesOverColumns< Lanes, TileSize / Lanes >( A, k, B, storer, 0 );
               };
               if( host && k <= 1 )
                  multiply( std::integral_constant< int, 1 >{} );
               else if( host && k <= 2 )
                  multiply( std::integral_constant< int, 2 >{} );
               else if( host && k <= 4 )
                  multiply( std::integral_constant< int, 4 >{} );
               else if( k <= 8 )
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
   auto multiply = [ & ]( auto tileSize )
   {
      Algorithms::Segments::multiReduceAllSegmentsInTiles< decltype( tileSize )::value >(
         AView.getSegments(),
         static_cast< int >( k ),
         SparseDenseMatrixProductFetch< decltype( AView ), BAccess >( AView, B, 0 ),
         TNL::Plus{},
         storer,
         ComputeRealType{ 0 },
         config );
   };

   // On the host, the multireduction with column-major B computes every tile of the columns for all rows before the
   // next one. The tile is the smallest power of two not smaller than k, so that the inner loop over the columns of the
   // tile has a constant trip count. The columns of B of one tile are read repeatedly by the rows, so they should fit
   // in the last-level cache, otherwise smaller tiles (down to 4 columns) are faster although they read A more times.
   if constexpr( Algorithms::Segments::isCSRSegments_v< SegmentsType >
                 && ( std::is_same_v< DeviceType, Devices::Host > || std::is_same_v< DeviceType, Devices::Sequential > )
                 && Organization == Algorithms::Segments::ColumnMajorOrder )
   {
      if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Default ) {
         static const std::size_t cacheSize = std::size_t( getCPUCacheSizes().L3 ) * 1024;
         const std::size_t columnSize = A.getColumns() * sizeof( BRealType );
         int tileSize = MaxTileSize;
         while( tileSize > 1 && tileSize / 2 >= k )
            tileSize /= 2;
         while( cacheSize > 0 && tileSize > 4 && tileSize * columnSize > cacheSize )
            tileSize /= 2;
         auto multiplyBy = [ & ]( auto size )
         {
            constexpr int Size = decltype( size )::value;
            if constexpr( Size < MaxTileSize ) {
               if( tileSize == Size ) {
                  multiply( size );
                  return true;
               }
            }
            return false;
         };
         if( multiplyBy( std::integral_constant< int, 1 >{} ) || multiplyBy( std::integral_constant< int, 2 >{} )
             || multiplyBy( std::integral_constant< int, 4 >{} ) || multiplyBy( std::integral_constant< int, 8 >{} )
             || multiplyBy( std::integral_constant< int, 16 >{} ) || multiplyBy( std::integral_constant< int, 32 >{} ) )
            return;
      }
   }
   multiply( std::integral_constant< int, MaxTileSize >{} );
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
   detail::checkSparseDenseMatrixProductDimensions( A, B, C );
   using ComputeRealType = typename SparseMatrix::ComputeRealType;
   const auto BView = B.getConstView();
   auto CView = C.getView();
   detail::sparseDenseMatrixProduct< MaxTileSize, DenseMatrixB::getOrganization() >(
      A,
      B.getColumns(),
      detail::DenseMatrixBAccess< decltype( BView ) >{ BView },
      detail::SparseDenseMatrixProductStorer< ComputeRealType, decltype( CView ), Real >{ CView, 0, alpha, beta },
      launchConfig );
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
   detail::checkSparseDenseMatrixProductDimensions( A, B, C );
   using ComputeRealType = typename SparseMatrix::ComputeRealType;
   const auto BView = B.getConstView();
   auto CView = C.getView();
   detail::sparseDenseMatrixProductLanesOverColumns< Lanes, ColumnsPerLane >(
      A,
      B.getColumns(),
      detail::DenseMatrixBAccess< decltype( BView ) >{ BView },
      detail::SparseDenseMatrixProductStorer< ComputeRealType, decltype( CView ), Real >{ CView, 0, alpha, beta },
      maxRowLength );
}

template<
   Algorithms::Segments::ElementsOrganization Organization,
   int MaxTileSize,
   typename SparseMatrix,
   typename FetchB,
   typename StoreC >
void
sparseDenseMatrixProductWithLambdas(
   const SparseMatrix& A,
   typename SparseMatrix::IndexType k,
   FetchB&& fetchB,
   StoreC&& storeC,
   const Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   using IndexType = typename SparseMatrix::IndexType;
   if( k < 0 )
      throw std::invalid_argument( "sparseDenseMatrixProductWithLambdas: the number of columns of B is negative" );
   if constexpr( is_dense_matrix_v< std::decay_t< FetchB > > ) {
      // B given by a dense matrix is read with the vectorized loads in its own organization
      if( fetchB.getRows() != A.getColumns() || fetchB.getColumns() != k )
         throw std::invalid_argument( "sparseDenseMatrixProductWithLambdas: wrong dimensions of B" );
      const auto BView = fetchB.getConstView();
      detail::sparseDenseMatrixProduct< MaxTileSize, std::decay_t< FetchB >::getOrganization() >(
         A, k, detail::DenseMatrixBAccess< decltype( BView ) >{ BView }, storeC, launchConfig );
   }
   else if constexpr( is_direct_fetch_v< FetchB > ) {
      // the organization of the array given by DirectFetch selects the kernel
      using DirectFetchType = std::decay_t< FetchB >;
      detail::sparseDenseMatrixProduct< MaxTileSize, DirectFetchType::getOrganization() >(
         A, k, detail::DirectBAccess< DirectFetchType >{ fetchB }, storeC, launchConfig );
   }
   else
      detail::sparseDenseMatrixProduct< MaxTileSize, Organization >(
         A, k, detail::LambdaBAccess< std::decay_t< FetchB >, IndexType >{ fetchB }, storeC, launchConfig );
}

template<
   int Lanes,
   int ColumnsPerLane,
   Algorithms::Segments::ElementsOrganization Organization,
   typename SparseMatrix,
   typename FetchB,
   typename StoreC >
void
sparseDenseMatrixProductWithLambdasLanesOverColumns(
   const SparseMatrix& A,
   typename SparseMatrix::IndexType k,
   FetchB&& fetchB,
   StoreC&& storeC,
   typename SparseMatrix::IndexType maxRowLength )
{
   using IndexType = typename SparseMatrix::IndexType;
   if( k < 0 )
      throw std::invalid_argument(
         "sparseDenseMatrixProductWithLambdasLanesOverColumns: the number of columns of B is negative" );
   if constexpr( is_dense_matrix_v< std::decay_t< FetchB > > ) {
      if( fetchB.getRows() != A.getColumns() || fetchB.getColumns() != k )
         throw std::invalid_argument( "sparseDenseMatrixProductWithLambdasLanesOverColumns: wrong dimensions of B" );
      const auto BView = fetchB.getConstView();
      detail::sparseDenseMatrixProductLanesOverColumns< Lanes, ColumnsPerLane >(
         A, k, detail::DenseMatrixBAccess< decltype( BView ) >{ BView }, storeC, maxRowLength );
   }
   else if constexpr( is_direct_fetch_v< FetchB > )
      detail::sparseDenseMatrixProductLanesOverColumns< Lanes, ColumnsPerLane >(
         A, k, detail::DirectBAccess< std::decay_t< FetchB > >{ fetchB }, storeC, maxRowLength );
   else
      detail::sparseDenseMatrixProductLanesOverColumns< Lanes, ColumnsPerLane >(
         A, k, detail::LambdaBAccess< std::decay_t< FetchB >, IndexType >{ fetchB }, storeC, maxRowLength );
}

namespace detail {

// The access to the elements of B^T for the dense matrix B, B^T( column, j ) = B( j, column ). The consecutive columns
// of B^T are consecutive in memory for column-major B, they are then loaded by vectorized loads.
template< typename DenseMatrixBView >
struct TransposedDenseMatrixBAccess
{
   using IndexType = typename DenseMatrixBView::IndexType;
   using RealType = std::remove_const_t< typename DenseMatrixBView::RealType >;

   static constexpr bool consecutive = DenseMatrixBView::getOrganization() == Algorithms::Segments::ColumnMajorOrder;

   __cuda_callable__
   RealType
   operator()( IndexType column, IndexType j ) const
   {
      return B( j, column );
   }

   template< int Count >
   __cuda_callable__
   void
   load( IndexType column, IndexType firstJ, RealType ( &b )[ Count ], bool vectorLoads ) const
   {
      if constexpr( consecutive )
         loadConsecutive< Count >( &B( firstJ, column ), b, vectorLoads );
      else
         for( int c = 0; c < Count; c++ )
            b[ c ] = B( firstJ + c, column );
   }

   // Tells whether W consecutive rows of B starting at multiples of W can be loaded by the vectorized loads, i.e. whether
   // B is column-major with the columns aligned to the vector width.
   template< int W >
   [[nodiscard]] bool
   canLoadVectors() const
   {
      if constexpr( consecutive ) {
         const auto& segments = B.getSegments();
         const auto address = reinterpret_cast< std::uintptr_t >( B.getValues().getData() );
         return address % ( W * sizeof( RealType ) ) == 0
             && ( B.getColumns() < 2 || ( segments.getGlobalIndex( 0, 1 ) - segments.getGlobalIndex( 0, 0 ) ) % W == 0 );
      }
      return false;
   }

   DenseMatrixBView B;
};

// The storer of the multireduction computing C^T = alpha * A * B^T + beta * C^T, i.e. the element C( j, row ).
template< typename ComputeRealType, typename DenseMatrixCView, typename Real >
struct TransposedProductStorer
{
   using IndexType = typename DenseMatrixCView::IndexType;

   __cuda_callable__
   void
   operator()( IndexType row, int j, const ComputeRealType& value ) const
   {
      auto& element = C( j, row );
      if( beta == Real{ 0 } )
         element = alpha * value;
      else
         element = alpha * value + beta * element;
   }

   // mutable, because the operator is const and the view gives a reference to its element only when not const
   mutable DenseMatrixCView C;
   Real alpha;
   Real beta;
};

}  // namespace detail

template< int MaxTileSize, typename DenseMatrixB, typename SparseMatrix, typename DenseMatrixC, typename Real >
void
denseSparseMatrixProduct(
   const DenseMatrixB& B,
   const SparseMatrix& A,
   DenseMatrixC& C,
   Real alpha,
   Real beta,
   TransposeState transposeA,
   const Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   static_assert( ! SparseMatrix::isSymmetric(), "denseSparseMatrixProduct does not support symmetric matrices yet." );
   using IndexType = typename SparseMatrix::IndexType;
   using DeviceType = typename SparseMatrix::DeviceType;
   using ComputeRealType = typename SparseMatrix::ComputeRealType;
   const IndexType opARows = transposeA == TransposeState::Transpose ? A.getColumns() : A.getRows();
   const IndexType opAColumns = transposeA == TransposeState::Transpose ? A.getRows() : A.getColumns();
   if( B.getColumns() != opARows )
      throw std::invalid_argument(
         "denseSparseMatrixProduct: the number of columns of B does not match the number of rows of op( A )" );
   if( C.getRows() != B.getRows() )
      throw std::invalid_argument( "denseSparseMatrixProduct: the number of rows of B does not match the number of rows of C" );
   if( C.getColumns() != opAColumns )
      throw std::invalid_argument(
         "denseSparseMatrixProduct: the number of columns of op( A ) does not match the number of columns of C" );
   const auto BView = B.getConstView();
   auto CView = C.getView();

   if( transposeA == TransposeState::Transpose ) {
      // C^T = A * B^T is the reduction in the rows of A, the kernel is selected by the organization of B^T
      constexpr auto organization = DenseMatrixB::getOrganization() == Algorithms::Segments::RowMajorOrder
                                     ? Algorithms::Segments::ColumnMajorOrder
                                     : Algorithms::Segments::RowMajorOrder;
      detail::sparseDenseMatrixProduct< MaxTileSize, organization >(
         A,
         B.getRows(),
         detail::TransposedDenseMatrixBAccess< decltype( BView ) >{ BView },
         detail::TransposedProductStorer< ComputeRealType, decltype( CView ), Real >{ CView, alpha, beta },
         launchConfig );
      return;
   }

   using RealType = typename DenseMatrixC::RealType;
   if constexpr( ! std::is_arithmetic_v< RealType > ) {
      // there are no atomic additions of the other types (e.g. complex numbers), the product is computed with the
      // transposed matrix
      TNL::Matrices::SparseMatrix< typename SparseMatrix::RealType, DeviceType, IndexType > AT;
      AT.getTransposition( A );
      denseSparseMatrixProduct< MaxTileSize >( B, AT, C, alpha, beta, TransposeState::Transpose, launchConfig );
   }
   else {
      // every element of A is multiplied by a column of B and added to a column of C by atomic additions, the rows of B
      // are processed by consecutive threads
      if( beta == Real{ 0 } )
         C.getValues().setValue( 0 );
      else if( beta != Real{ 1 } )
         C.getValues() *= beta;
      const auto AView = A.getConstView();
      const IndexType k = B.getRows();
      Algorithms::parallelFor< DeviceType >(
         0,
         A.getRows() * k,
         [ = ] __cuda_callable__( IndexType index ) mutable
         {
            const IndexType row = index / k;
            const IndexType i = index % k;
            const auto aRow = AView.getRow( row );
            const ComputeRealType b = alpha * BView( i, row );
            for( IndexType l = 0; l < aRow.getSize(); l++ ) {
               const IndexType column = aRow.getColumnIndex( l );
               if( column == paddingIndex< IndexType > )
                  continue;
               Algorithms::AtomicOperations< DeviceType >::add( CView( i, column ), b * aRow.getValue( l ) );
            }
         } );
   }
}

}  // namespace TNL::Matrices
