// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

/*
 * Hand-written reference kernels for the product of a CSR sparse matrix and
 * a dense matrix, C = alpha * A * B + beta * C. They work on raw arrays so that
 * they can serve as a performance baseline for the generic implementation
 * based on TNL segments.
 *
 * Two families of kernels are implemented:
 *
 * - RowGroup: a group of TPR threads processes one row of A and every thread
 *   keeps a tile of K columns of C in registers. Each element of A is loaded
 *   once per tile and reused K times. For small k (K == k) this is a fused
 *   SpMV over k vectors, for large k the columns are processed in tiles
 *   (grid dimension y). Works with both row-major and column-major B and C.
 *   The mapping of threads to rows is the CSR vector kernel of SpMV [1]
 *   extended to a block of vectors as in [2].
 *
 * - LanesOverColumns: a group of G lanes processes one row of A, the lanes
 *   load a chunk of G elements of the row cooperatively and broadcast them
 *   by warp shuffles, and every lane computes V consecutive columns of C
 *   with vectorized loads of B. Requires row-major B and C. This is the
 *   row-splitting SpMM with the row-major coalesced access to B [3], with
 *   the cooperative loading of the row (coalesced row caching, here in
 *   registers instead of shared memory) and several columns per lane
 *   (coarse-grained warp merging) of [4], and the subwarp tiling and vector
 *   loads of [5].
 *
 *   RowGroup and LanesOverColumns can process the rows in a given order
 *   (CSRArrays::rowOrder), e.g. sorted by their lengths, so that the groups
 *   of one warp process similarly long rows (row swizzle of [5]).
 *
 * - NnzSplit: like LanesOverColumns, but the groups of G lanes process chunks
 *   of the same number of nonzero elements regardless of the row boundaries,
 *   and the rows split between several chunks are accumulated atomically.
 *   This balances the work for matrices with very irregular row lengths.
 *   It is a simplified variant of the merge-based SpMM [3], which partitions
 *   the work by the merge path of the rows and the nonzero elements [6].
 *   Here only the nonzero elements are partitioned and the empty rows are
 *   processed by the chunk containing their position. Requires row-major B
 *   and C.
 *
 * - Binned: the rows sorted by their lengths are divided into bins and every
 *   bin is processed by the RowGroup kernel with the number of threads per
 *   row adequate to the lengths of its rows, like the binning of rows in the
 *   adaptive CSR SpMV [7]. The very long rows are split into pieces processed
 *   by whole warps, which add their results atomically. Works with both
 *   row-major and column-major B and C.
 *
 * References:
 *
 * [1] N. Bell, M. Garland: Implementing Sparse Matrix-Vector Multiplication
 *     on Throughput-Oriented Processors, SC 2009.
 * [2] H. Anzt, S. Tomov, J. Dongarra: Accelerating the LOBPCG Method on GPUs
 *     Using a Blocked Sparse Matrix Vector Product, SpringSim (HPC) 2015.
 * [3] C. Yang, A. Buluç, J. D. Owens: Design Principles for Sparse Matrix
 *     Multiplication on the GPU, Euro-Par 2018, LNCS 11014, pp. 672-687.
 *     arXiv:1803.08601.
 * [4] G. Huang, G. Dai, Y. Wang, H. Yang: GE-SpMM: General-Purpose Sparse
 *     Matrix-Matrix Multiplication on GPUs for Graph Neural Networks,
 *     SC 2020. arXiv:2007.03179.
 * [5] T. Gale, M. Zaharia, C. Young, E. Elsen: Sparse GPU Kernels for Deep
 *     Learning, SC 2020. arXiv:2006.10901.
 * [6] D. Merrill, M. Garland: Merge-Based Parallel Sparse Matrix-Vector
 *     Multiplication, SC 2016.
 * [7] A. Ashari, N. Sedaghati, J. Eisenlohr, S. Parthasarathy, P. Sadayappan:
 *     Fast Sparse Matrix-Vector Multiplication on GPUs for Graph
 *     Applications, SC 2014.
 */

#pragma once

#include <TNL/Backend/Functions.h>
#include <TNL/Backend/KernelLaunch.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/Math.h>

namespace TNL::Benchmarks::SpMM {

//! \brief Raw arrays of a CSR matrix.
template< typename Real, typename Index >
struct CSRArrays
{
   const Index* rowPointers = nullptr;
   const Index* columnIndexes = nullptr;
   const Real* values = nullptr;
   Index rows = 0;
   Index columns = 0;
   Index nonzeros = 0;
   //! Optional permutation of the rows giving the order in which they are processed.
   const Index* rowOrder = nullptr;
};

//! \brief Raw array of a dense matrix with a leading dimension.
template< typename Real, typename Index >
struct DenseArray
{
   Real* data = nullptr;
   Index rows = 0;
   Index columns = 0;
   //! Distance between rows (row-major) or columns (column-major).
   Index leadingDimension = 0;
};

#if defined( __CUDACC__ ) || defined( __HIP__ )

namespace detail {

template< bool RowMajor, typename Index >
__cuda_callable__
Index
denseIndex( Index row, Index column, Index leadingDimension )
{
   if constexpr( RowMajor )
      return row * leadingDimension + column;
   else
      return column * leadingDimension + row;
}

// Native vector types for vectorized loads/stores of N consecutive values.
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

// Loads N consecutive values. The source must be aligned to the vector width
// returned by vectorWidth< Real, N >() when `aligned` is true.
template< int N, typename Real >
__device__
void
loadConsecutive( const Real* __restrict__ source, Real ( &target )[ N ], bool aligned )
{
   constexpr int W = vectorWidth< Real, N >();
   if constexpr( W > 1 ) {
      if( aligned ) {
         using Vector = typename VectorType< Real, W >::type;
         const Vector* vectorSource = reinterpret_cast< const Vector* >( source );
   #pragma unroll
         for( int i = 0; i < N / W; i++ ) {
            const Vector v = vectorSource[ i ];
            const Real* scalars = reinterpret_cast< const Real* >( &v );
   #pragma unroll
            for( int j = 0; j < W; j++ )
               target[ i * W + j ] = scalars[ j ];
         }
         return;
      }
   }
   #pragma unroll
   for( int i = 0; i < N; i++ )
      target[ i ] = source[ i ];
}

template< typename T >
__device__
T
warpShuffleIndex( T value, int sourceLane, int width )
{
   #if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   return __shfl_sync( Backend::getWarpFullMask(), value, sourceLane, width );
   #else
   return value;
   #endif
}

template< typename T >
__device__
T
warpReduceMax( T value )
{
   #pragma unroll
   for( int offset = Backend::getWarpSize() / 2; offset > 0; offset /= 2 )
      value = max( value, Backend::warp_shuffle_xor( value, offset ) );
   return value;
}

/*
 * RowGroup kernel: TPR threads per row, a tile of K columns of C per thread.
 *
 * The kernel processes the slots [firstSlot, lastSlot). The slot is mapped to
 * the row by rowOrder (identity when it is nullptr). When pieceBegins and
 * pieceEnds are given, the slot processes only the elements [pieceBegins[slot],
 * pieceEnds[slot]) of the row and adds the result atomically to C, which must
 * be multiplied by beta in advance.
 *
 * The slots are traversed by a grid-stride loop whose trip count is uniform
 * within each warp, so that the warp shuffles are always executed by the
 * whole warp. blockIdx.y selects the tile of columns.
 */
template< int TPR, int K, bool RowMajor, typename Real, typename Index >
__global__
void
spmmRowGroupKernel(
   const Index* __restrict__ rowPointers,
   const Index* __restrict__ columnIndexes,
   const Real* __restrict__ values,
   const Index* __restrict__ rowOrder,
   const Index* __restrict__ pieceBegins,
   const Index* __restrict__ pieceEnds,
   const Real* __restrict__ B,
   Index ldb,
   Real* __restrict__ C,
   Index ldc,
   Index firstSlot,
   Index lastSlot,
   Index firstTile,
   Index k,
   Real alpha,
   Real beta,
   bool alignedB )
{
   #if defined( __CUDACC__ ) || defined( __HIP__ )
   static_assert( TPR <= Backend::getMinWarpSize() );
   constexpr int warpSize = Backend::getWarpSize();
   constexpr int groupsPerWarp = warpSize / TPR;

   const Index threadIdx_ = static_cast< Index >( blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index warpIdx = threadIdx_ / warpSize;
   const Index warpsCount = static_cast< Index >( gridDim.x ) * blockDim.x / warpSize;
   const int laneInWarp = threadIdx.x % warpSize;
   const int groupInWarp = laneInWarp / TPR;
   const int lane = laneInWarp % TPR;

   const Index columnOffset = ( firstTile + static_cast< Index >( blockIdx.y ) ) * K;
   const bool fullTile = columnOffset + K <= k;
   const Real* tileB = B + denseIndex< RowMajor >( Index( 0 ), columnOffset, ldb );

   const bool pieces = pieceBegins != nullptr;
   for( Index firstRow = firstSlot + warpIdx * groupsPerWarp; firstRow < lastSlot; firstRow += warpsCount * groupsPerWarp ) {
      const Index slot = firstRow + groupInWarp;
      const Index row = ( rowOrder != nullptr && slot < lastSlot ) ? rowOrder[ slot ] : slot;
      Real accumulator[ K ];
      #pragma unroll
      for( int c = 0; c < K; c++ )
         accumulator[ c ] = 0;

      if( slot < lastSlot ) {
         const Index begin = pieces ? pieceBegins[ slot ] : rowPointers[ row ];
         const Index end = pieces ? pieceEnds[ slot ] : rowPointers[ row + 1 ];
         for( Index i = begin + lane; i < end; i += TPR ) {
            const Index column = columnIndexes[ i ];
            const Real value = values[ i ];
            Real b[ K ];
            if constexpr( RowMajor ) {
               if( fullTile )
                  loadConsecutive< K >( tileB + column * ldb, b, alignedB );
               else {
      #pragma unroll
                  for( int c = 0; c < K; c++ )
                     b[ c ] = ( columnOffset + c < k ) ? tileB[ column * ldb + c ] : Real( 0 );
               }
            }
            else {
      #pragma unroll
               for( int c = 0; c < K; c++ )
                  b[ c ] = ( fullTile || columnOffset + c < k ) ? tileB[ c * ldb + column ] : Real( 0 );
            }
      #pragma unroll
            for( int c = 0; c < K; c++ )
               accumulator[ c ] += value * b[ c ];
         }
      }

      // butterfly reduction, all lanes of the group end up with the full sums
      if constexpr( TPR > 1 ) {
      #pragma unroll
         for( int offset = TPR / 2; offset > 0; offset /= 2 ) {
      #pragma unroll
            for( int c = 0; c < K; c++ )
               accumulator[ c ] += Backend::warp_shuffle_xor( accumulator[ c ], offset );
         }
      }

      if( slot < lastSlot ) {
            // the lanes of the group share the writes of the tile
      #pragma unroll
         for( int c = 0; c < K; c++ ) {
            if( c % TPR == lane && ( fullTile || columnOffset + c < k ) ) {
               const Index idx = denseIndex< RowMajor >( row, columnOffset + c, ldc );
               if( pieces )
                  atomicAdd( C + idx, alpha * accumulator[ c ] );
               else if( beta == Real( 0 ) )
                  C[ idx ] = alpha * accumulator[ c ];
               else
                  C[ idx ] = alpha * accumulator[ c ] + beta * C[ idx ];
            }
         }
      }
   }
   #endif
}

/*
 * LanesOverColumns kernel: G lanes per row, V consecutive columns per lane,
 * row-major B and C. blockIdx.y selects the tile of G * V columns.
 *
 * The elements of the row are loaded in chunks of G (one element per lane)
 * and broadcast within the group by warp shuffles. The number of chunks is
 * the maximum over the groups of the warp, so that the shuffles are always
 * executed by the whole warp.
 */
template< int G, int V, typename Real, typename Index >
__global__
void
spmmLanesOverColumnsKernel(
   const Index* __restrict__ rowPointers,
   const Index* __restrict__ columnIndexes,
   const Real* __restrict__ values,
   const Index* __restrict__ rowOrder,
   const Real* __restrict__ B,
   Index ldb,
   Real* __restrict__ C,
   Index ldc,
   Index rows,
   Index k,
   Real alpha,
   Real beta,
   bool alignedB )
{
   #if defined( __CUDACC__ ) || defined( __HIP__ )
   static_assert( G <= Backend::getMinWarpSize() );
   constexpr int warpSize = Backend::getWarpSize();
   constexpr int groupsPerWarp = warpSize / G;

   const Index threadIdx_ = static_cast< Index >( blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index warpIdx = threadIdx_ / warpSize;
   const Index warpsCount = static_cast< Index >( gridDim.x ) * blockDim.x / warpSize;
   const int laneInWarp = threadIdx.x % warpSize;
   const int groupInWarp = laneInWarp / G;
   const int lane = laneInWarp % G;

   const Index firstColumn = static_cast< Index >( blockIdx.y ) * G * V + lane * V;
   const bool fullColumns = firstColumn + V <= k;
   const bool anyColumn = firstColumn < k;

   for( Index firstRow = warpIdx * groupsPerWarp; firstRow < rows; firstRow += warpsCount * groupsPerWarp ) {
      const Index slot = firstRow + groupInWarp;
      const Index row = ( rowOrder != nullptr && slot < rows ) ? rowOrder[ slot ] : slot;
      Index begin = 0;
      Index length = 0;
      if( slot < rows ) {
         begin = rowPointers[ row ];
         length = rowPointers[ row + 1 ] - begin;
      }
      const Index maxLength = ( G == warpSize ) ? length : warpReduceMax( length );

      Real accumulator[ V ];
      #pragma unroll
      for( int c = 0; c < V; c++ )
         accumulator[ c ] = 0;

      for( Index chunk = 0; chunk < maxLength; chunk += G ) {
         Index myColumn = 0;
         Real myValue = 0;
         if( chunk + lane < length ) {
            myColumn = columnIndexes[ begin + chunk + lane ];
            myValue = values[ begin + chunk + lane ];
         }
      #pragma unroll
         for( int j = 0; j < G; j++ ) {
            const Index column = warpShuffleIndex( myColumn, j, G );
            const Real value = warpShuffleIndex( myValue, j, G );
            if( chunk + j < length && anyColumn ) {
               const Real* rowB = B + column * ldb + firstColumn;
               Real b[ V ];
               if( fullColumns )
                  loadConsecutive< V >( rowB, b, alignedB );
               else {
      #pragma unroll
                  for( int c = 0; c < V; c++ )
                     b[ c ] = ( firstColumn + c < k ) ? rowB[ c ] : Real( 0 );
               }
      #pragma unroll
               for( int c = 0; c < V; c++ )
                  accumulator[ c ] += value * b[ c ];
            }
         }
      }

      if( slot < rows ) {
      #pragma unroll
         for( int c = 0; c < V; c++ ) {
            if( firstColumn + c < k ) {
               const Index idx = row * ldc + firstColumn + c;
               if( beta == Real( 0 ) )
                  C[ idx ] = alpha * accumulator[ c ];
               else
                  C[ idx ] = alpha * accumulator[ c ] + beta * C[ idx ];
            }
         }
      }
   }
   #endif
}

// Returns the first index i in [0, size) with array[i] >= value, or size if there is none.
template< typename Index >
__device__
Index
lowerBound( const Index* __restrict__ array, Index size, Index value )
{
   Index first = 0;
   while( size > 0 ) {
      const Index half = size / 2;
      if( array[ first + half ] < value ) {
         first += half + 1;
         size -= half + 1;
      }
      else
         size = half;
   }
   return first;
}

// Index of the row containing the element at the given position, i.e. the last
// row whose first element is at the position or before it.
template< typename Index >
__device__
Index
rowOfElement( const Index* __restrict__ rowPointers, Index rows, Index position )
{
   return lowerBound( rowPointers, rows, position + 1 ) - 1;
}

/*
 * Preparation for the NnzSplit kernel: the rows of C which are split between
 * several chunks are multiplied by beta, so that the chunks can add their
 * contributions atomically. Every row is processed by the first chunk boundary
 * inside it.
 */
template< typename Real, typename Index >
__global__
void
spmmNnzSplitPrepareKernel(
   const Index* __restrict__ rowPointers,
   Real* __restrict__ C,
   Index ldc,
   Index rows,
   Index k,
   Index chunkSize,
   Index chunks,
   Real beta )
{
   const Index stride = static_cast< Index >( gridDim.x ) * blockDim.x;
   for( Index chunk = static_cast< Index >( blockIdx.x ) * blockDim.x + threadIdx.x + 1; chunk < chunks; chunk += stride ) {
      const Index position = chunk * chunkSize;
      const Index row = rowOfElement( rowPointers, rows, position );
      const Index begin = rowPointers[ row ];
      if( begin < position && begin >= position - chunkSize )
         for( Index c = 0; c < k; c++ )
            C[ row * ldc + c ] = ( beta == Real( 0 ) ) ? Real( 0 ) : beta * C[ row * ldc + c ];
   }
}

/*
 * NnzSplit kernel: G lanes per chunk of the nonzero elements, V consecutive
 * columns per lane, row-major B and C. blockIdx.y selects the tile of G * V
 * columns.
 *
 * The nonzero elements are split into chunks of `chunkSize` elements regardless
 * of the row boundaries, so that the work is balanced even for matrices with
 * very long rows. A chunk owns the rows which begin inside it (and the last
 * chunk also the empty rows at the end of the matrix). The rows which are
 * completely inside one chunk are stored directly, the contributions to the
 * other rows are added atomically to C prepared by spmmNnzSplitPrepareKernel.
 * The elements are loaded and broadcast as in the LanesOverColumns kernel.
 * The chunk size is a multiple of G, so that all groups of a warp execute the
 * same number of warp shuffles.
 */
template< int G, int V, typename Real, typename Index >
__global__
void
spmmNnzSplitKernel(
   const Index* __restrict__ rowPointers,
   const Index* __restrict__ columnIndexes,
   const Real* __restrict__ values,
   const Real* __restrict__ B,
   Index ldb,
   Real* __restrict__ C,
   Index ldc,
   Index rows,
   Index k,
   Index nonzeros,
   Index chunkSize,
   Index chunks,
   Real alpha,
   Real beta,
   bool alignedB )
{
   #if defined( __CUDACC__ ) || defined( __HIP__ )
   static_assert( G <= Backend::getMinWarpSize() );
   constexpr int warpSize = Backend::getWarpSize();
   constexpr int groupsPerWarp = warpSize / G;

   const Index threadIdx_ = static_cast< Index >( blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index warpIdx = threadIdx_ / warpSize;
   const Index warpsCount = static_cast< Index >( gridDim.x ) * blockDim.x / warpSize;
   const int laneInWarp = threadIdx.x % warpSize;
   const int groupInWarp = laneInWarp / G;
   const int lane = laneInWarp % G;

   const Index firstColumn = static_cast< Index >( blockIdx.y ) * G * V + lane * V;
   const bool fullColumns = firstColumn + V <= k;
   const bool anyColumn = firstColumn < k;

   Real accumulator[ V ];
   Index row = 0;
   Index rowBegin = 0;
   Index rowEnd = 0;
   Index begin = 0;
   Index end = 0;

   // stores the accumulated part of the current row and resets the accumulator
   auto flush = [ & ]()
   {
      const bool split = rowBegin < begin || rowEnd > end;
      #pragma unroll
      for( int c = 0; c < V; c++ ) {
         if( firstColumn + c < k ) {
            const Index idx = row * ldc + firstColumn + c;
            if( split )
               atomicAdd( C + idx, alpha * accumulator[ c ] );
            else if( beta == Real( 0 ) )
               C[ idx ] = alpha * accumulator[ c ];
            else
               C[ idx ] = alpha * accumulator[ c ] + beta * C[ idx ];
         }
         accumulator[ c ] = 0;
      }
   };

   for( Index firstChunk = warpIdx * groupsPerWarp; firstChunk < chunks; firstChunk += warpsCount * groupsPerWarp ) {
      const Index chunk = firstChunk + groupInWarp;
      const bool active = chunk < chunks;
      begin = chunk * chunkSize;
      end = active ? min( begin + chunkSize, nonzeros ) : begin;
      if( active && rows > 0 ) {
         // the row split by the beginning of the chunk, or the first row owned by the chunk
         row = rowOfElement( rowPointers, rows, begin );
         if( rowPointers[ row ] == begin )
            row = lowerBound( rowPointers, rows, begin );
         rowBegin = rowPointers[ row ];
         rowEnd = rowPointers[ row + 1 ];
      }
      #pragma unroll
      for( int c = 0; c < V; c++ )
         accumulator[ c ] = 0;

      for( Index step = begin; step < begin + chunkSize; step += G ) {
         Index myColumn = 0;
         Real myValue = 0;
         if( step + lane < end ) {
            myColumn = columnIndexes[ step + lane ];
            myValue = values[ step + lane ];
         }
      #pragma unroll
         for( int j = 0; j < G; j++ ) {
            const Index column = warpShuffleIndex( myColumn, j, G );
            const Real value = warpShuffleIndex( myValue, j, G );
            if( step + j < end ) {
               while( step + j >= rowEnd ) {
                  flush();
                  row++;
                  rowBegin = rowEnd;
                  rowEnd = rowPointers[ row + 1 ];
               }
               if( anyColumn ) {
                  const Real* rowB = B + column * ldb + firstColumn;
                  Real b[ V ];
                  if( fullColumns )
                     loadConsecutive< V >( rowB, b, alignedB );
                  else {
      #pragma unroll
                     for( int c = 0; c < V; c++ )
                        b[ c ] = ( firstColumn + c < k ) ? rowB[ c ] : Real( 0 );
                  }
      #pragma unroll
                  for( int c = 0; c < V; c++ )
                     accumulator[ c ] += value * b[ c ];
               }
            }
         }
      }

      if( active ) {
         // the last row with elements in the chunk and the empty rows owned by the chunk
         const Index ownedEnd = ( chunk == chunks - 1 ) ? nonzeros + 1 : end;
         while( row < rows && rowBegin < ownedEnd ) {
            flush();
            row++;
            if( row < rows ) {
               rowBegin = rowEnd;
               rowEnd = rowPointers[ row + 1 ];
            }
         }
      }
   }
   #endif
}

// Number of blocks for the grid-stride loops: enough to fill the device
// several times, but no more than needed for one pass over the rows.
template< typename Index >
unsigned int
gridStrideBlocks( Index rows, int rowsPerBlock )
{
   constexpr Index maxBlocks = 1 << 16;
   const Index needed = roundUpDivision( rows, static_cast< Index >( rowsPerBlock ) );
   return static_cast< unsigned int >( max( Index( 1 ), min( needed, maxBlocks ) ) );
}

// Multiplies the rows rowOrder[0], ..., rowOrder[count - 1] of C by beta.
template< bool RowMajor, typename Real, typename Index >
__global__
void
spmmScaleRowsKernel( const Index* __restrict__ rowOrder, Index count, Real* __restrict__ C, Index ldc, Index k, Real beta )
{
   const Index stride = static_cast< Index >( gridDim.x ) * blockDim.x;
   for( Index i = static_cast< Index >( blockIdx.x ) * blockDim.x + threadIdx.x; i < count * k; i += stride ) {
      const Index idx = denseIndex< RowMajor >( rowOrder[ i / k ], i % k, ldc );
      C[ idx ] = ( beta == Real( 0 ) ) ? Real( 0 ) : beta * C[ idx ];
   }
}

// Launches the RowGroup kernel for the slots [firstSlot, lastSlot), see spmmRowGroupKernel. The tiles of
// columns are processed by consecutive launches of tilesPerLaunch tiles (all tiles at once when it is 0).
template< int TPR, int K, bool RowMajor, typename Real, typename Index >
void
launchRowGroup(
   const CSRArrays< Real, Index >& A,
   const DenseArray< Real, Index >& B,
   DenseArray< Real, Index >& C,
   const Index* rowOrder,
   const Index* pieceBegins,
   const Index* pieceEnds,
   Index firstSlot,
   Index lastSlot,
   Real alpha,
   Real beta,
   Index tilesPerLaunch = 0 )
{
   constexpr int blockSize = 256;
   const Index tiles = roundUpDivision( C.columns, static_cast< Index >( K ) );
   if( tilesPerLaunch <= 0 || tilesPerLaunch > tiles )
      tilesPerLaunch = tiles;
   if( tilesPerLaunch > static_cast< Index >( Backend::getMaxGridYSize() ) )
      throw std::invalid_argument( "spmmRowGroup: too many tiles of columns" );

   // the vectorized loads need every row of B and the tile offsets aligned to the vector width
   constexpr int W = vectorWidth< Real, K >();
   const bool alignedB = RowMajor && B.leadingDimension % W == 0;

   constexpr auto kernel = spmmRowGroupKernel< TPR, K, RowMajor, Real, Index >;
   for( Index firstTile = 0; firstTile < tiles; firstTile += tilesPerLaunch ) {
      Backend::LaunchConfiguration config;
      config.blockSize.x = blockSize;
      config.gridSize.x = gridStrideBlocks( lastSlot - firstSlot, blockSize / TPR );
      config.gridSize.y = min( tilesPerLaunch, tiles - firstTile );
      Backend::launchKernelAsync(
         kernel,
         config,
         A.rowPointers,
         A.columnIndexes,
         A.values,
         rowOrder,
         pieceBegins,
         pieceEnds,
         static_cast< const Real* >( B.data ),
         B.leadingDimension,
         C.data,
         C.leadingDimension,
         firstSlot,
         lastSlot,
         firstTile,
         C.columns,
         alpha,
         beta,
         alignedB );
   }
}

}  // namespace detail

/**
 * \brief Computes C = alpha * A * B + beta * C with the RowGroup kernel.
 *
 * \tparam TPR is the number of threads per row of A (a power of two up to 32).
 * \tparam K is the number of columns of C computed by each thread.
 * \tparam RowMajor tells whether B and C are stored in row-major order.
 * \param tilesPerLaunch is the number of tiles of columns processed by one kernel launch, the launches
 *        follow each other (0 means all tiles in one launch). Fewer tiles per launch reduce the working
 *        set of B in the cache, which matters for the column-major layout.
 */
template< int TPR, int K, bool RowMajor, typename Real, typename Index >
void
spmmRowGroup(
   const CSRArrays< Real, Index >& A,
   const DenseArray< Real, Index >& B,
   DenseArray< Real, Index >& C,
   Real alpha = 1,
   Real beta = 0,
   Index tilesPerLaunch = 0 )
{
   detail::launchRowGroup< TPR, K, RowMajor >(
      A,
      B,
      C,
      A.rowOrder,
      static_cast< const Index* >( nullptr ),
      static_cast< const Index* >( nullptr ),
      Index( 0 ),
      A.rows,
      alpha,
      beta,
      tilesPerLaunch );
   Backend::streamSynchronize( Backend::LaunchConfiguration{}.stream );
}

/**
 * \brief Partition of the rows of a CSR matrix for the Binned kernel.
 *
 * The rows are sorted by their lengths in descending order (`rowOrder`). The
 * first `longRows` rows are split into pieces of at most `pieceSize` elements,
 * the piece `p` covers the elements `[pieceBegins[p], pieceEnds[p])` of the row
 * `pieceRows[p]`. The other rows are divided into bins, the slots
 * `[binBegins[b], binBegins[b + 1])` of `rowOrder` are processed with
 * `32 >> b` threads per row. All arrays except `binBegins` are on the device.
 */
template< typename Index >
struct RowBins
{
   static constexpr int bins = 6;

   const Index* rowOrder = nullptr;
   Index longRows = 0;
   const Index* pieceRows = nullptr;
   const Index* pieceBegins = nullptr;
   const Index* pieceEnds = nullptr;
   Index pieces = 0;
   Index binBegins[ bins + 1 ] = {};
};

/**
 * \brief Computes C = alpha * A * B + beta * C with the Binned kernel.
 *
 * The rows of every bin are processed by the RowGroup kernel with the number
 * of threads per row adequate to their lengths, and the pieces of the long
 * rows are processed by warps which add their results atomically to C.
 *
 * \tparam K is the number of columns of C computed by each thread.
 * \tparam RowMajor tells whether B and C are stored in row-major order.
 */
template< int K, bool RowMajor, typename Real, typename Index >
void
spmmBinned(
   const CSRArrays< Real, Index >& A,
   const RowBins< Index >& bins,
   const DenseArray< Real, Index >& B,
   DenseArray< Real, Index >& C,
   Real alpha = 1,
   Real beta = 0 )
{
   if( bins.longRows > 0 ) {
      constexpr int blockSize = 256;
      Backend::LaunchConfiguration config;
      config.blockSize.x = blockSize;
      config.gridSize.x = detail::gridStrideBlocks( bins.longRows * C.columns, blockSize );
      constexpr auto kernel = detail::spmmScaleRowsKernel< RowMajor, Real, Index >;
      Backend::launchKernelAsync( kernel, config, bins.rowOrder, bins.longRows, C.data, C.leadingDimension, C.columns, beta );
      detail::launchRowGroup< 32, K, RowMajor >(
         A, B, C, bins.pieceRows, bins.pieceBegins, bins.pieceEnds, Index( 0 ), bins.pieces, alpha, beta );
   }

   auto launchBin = [ & ]( auto TPR_, int bin )
   {
      constexpr int TPR = decltype( TPR_ )::value;
      if( bins.binBegins[ bin ] < bins.binBegins[ bin + 1 ] )
         detail::launchRowGroup< TPR, K, RowMajor >(
            A,
            B,
            C,
            bins.rowOrder,
            static_cast< const Index* >( nullptr ),
            static_cast< const Index* >( nullptr ),
            bins.binBegins[ bin ],
            bins.binBegins[ bin + 1 ],
            alpha,
            beta );
   };
   launchBin( std::integral_constant< int, 32 >{}, 0 );
   launchBin( std::integral_constant< int, 16 >{}, 1 );
   launchBin( std::integral_constant< int, 8 >{}, 2 );
   launchBin( std::integral_constant< int, 4 >{}, 3 );
   launchBin( std::integral_constant< int, 2 >{}, 4 );
   launchBin( std::integral_constant< int, 1 >{}, 5 );
   Backend::streamSynchronize( Backend::LaunchConfiguration{}.stream );
}

/**
 * \brief Computes C = alpha * A * B + beta * C with the LanesOverColumns kernel.
 *
 * B and C must be stored in row-major order.
 *
 * \tparam G is the number of lanes per row of A (a power of two up to 32).
 * \tparam V is the number of consecutive columns of C computed by each lane.
 */
template< int G, int V, typename Real, typename Index >
void
spmmLanesOverColumns(
   const CSRArrays< Real, Index >& A,
   const DenseArray< Real, Index >& B,
   DenseArray< Real, Index >& C,
   Real alpha = 1,
   Real beta = 0 )
{
   constexpr int blockSize = 256;
   Backend::LaunchConfiguration config;
   config.blockSize.x = blockSize;
   config.gridSize.x = detail::gridStrideBlocks( A.rows, blockSize / G );
   config.gridSize.y = roundUpDivision( C.columns, static_cast< Index >( G * V ) );
   if( config.gridSize.y > static_cast< unsigned int >( Backend::getMaxGridYSize() ) )
      throw std::invalid_argument( "spmmLanesOverColumns: too many tiles of columns" );

   // every lane starts at a multiple of V, so it suffices to check the rows of B
   constexpr int W = detail::vectorWidth< Real, V >();
   const bool alignedB = B.leadingDimension % W == 0;

   constexpr auto kernel = detail::spmmLanesOverColumnsKernel< G, V, Real, Index >;
   Backend::launchKernelSync(
      kernel,
      config,
      A.rowPointers,
      A.columnIndexes,
      A.values,
      A.rowOrder,
      static_cast< const Real* >( B.data ),
      B.leadingDimension,
      C.data,
      C.leadingDimension,
      A.rows,
      C.columns,
      alpha,
      beta,
      alignedB );
}

/**
 * \brief Computes C = alpha * A * B + beta * C with the NnzSplit kernel.
 *
 * B and C must be stored in row-major order.
 *
 * \tparam G is the number of lanes per chunk of nonzero elements (a power of two up to 32).
 * \tparam V is the number of consecutive columns of C computed by each lane.
 * \param chunkSize is the number of nonzero elements per chunk, a positive multiple of G.
 */
template< int G, int V, typename Real, typename Index >
void
spmmNnzSplit(
   const CSRArrays< Real, Index >& A,
   const DenseArray< Real, Index >& B,
   DenseArray< Real, Index >& C,
   Index chunkSize,
   Real alpha = 1,
   Real beta = 0 )
{
   if( chunkSize <= 0 || chunkSize % G != 0 )
      throw std::invalid_argument( "spmmNnzSplit: the chunk size must be a positive multiple of G" );
   const Index chunks = max( Index( 1 ), roundUpDivision( A.nonzeros, chunkSize ) );

   constexpr int blockSize = 256;
   if( chunks > 1 ) {
      Backend::LaunchConfiguration config;
      config.blockSize.x = blockSize;
      config.gridSize.x = detail::gridStrideBlocks( chunks - 1, blockSize );
      constexpr auto kernel = detail::spmmNnzSplitPrepareKernel< Real, Index >;
      Backend::launchKernelAsync(
         kernel, config, A.rowPointers, C.data, C.leadingDimension, A.rows, C.columns, chunkSize, chunks, beta );
   }

   Backend::LaunchConfiguration config;
   config.blockSize.x = blockSize;
   config.gridSize.x = detail::gridStrideBlocks( chunks, blockSize / G );
   config.gridSize.y = roundUpDivision( C.columns, static_cast< Index >( G * V ) );
   if( config.gridSize.y > static_cast< unsigned int >( Backend::getMaxGridYSize() ) )
      throw std::invalid_argument( "spmmNnzSplit: too many tiles of columns" );

   constexpr int W = detail::vectorWidth< Real, V >();
   const bool alignedB = B.leadingDimension % W == 0;

   constexpr auto kernel = detail::spmmNnzSplitKernel< G, V, Real, Index >;
   Backend::launchKernelSync(
      kernel,
      config,
      A.rowPointers,
      A.columnIndexes,
      A.values,
      static_cast< const Real* >( B.data ),
      B.leadingDimension,
      C.data,
      C.leadingDimension,
      A.rows,
      C.columns,
      A.nonzeros,
      chunkSize,
      chunks,
      alpha,
      beta,
      alignedB );
}

#endif

}  // namespace TNL::Benchmarks::SpMM
