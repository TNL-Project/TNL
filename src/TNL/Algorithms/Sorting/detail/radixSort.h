// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

#include <TNL/Backend.h>
#include <TNL/Containers/Array.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL::Algorithms::Sorting::detail {

//! \brief Number of bits of the key processed in one radix sort pass.
constexpr int radixSortBits = 8;

//! \brief Number of digit buckets, i.e. `2^radixSortBits`.
constexpr int radixSortBuckets = 1 << radixSortBits;

//! \brief Number of threads per block in the radix sort kernels.
constexpr int radixSortBlockSize = 256;

//! \brief Number of elements processed by one thread in one tile.
//!
//! Processing multiple elements per thread amortizes the shared-memory
//! zeroing, the per-digit scan and the synchronization over more elements
//! and provides more independent loads to hide the memory latency.
constexpr int radixSortItemsPerThread = 8;

//! \brief Number of elements processed by one tile of the kernels.
constexpr int radixSortTileSize = radixSortBlockSize * radixSortItemsPerThread;

// The per-digit phases of the scatter kernel rely on each thread handling
// exactly one digit, which requires the block size to be equal to the number
// of digit buckets.
static_assert( radixSortBlockSize == radixSortBuckets, "radixSortBlockSize must be equal to radixSortBuckets" );

//! \brief Maximum number of warps per block in the radix sort kernels,
//! assuming the smallest supported warp size of 32.
constexpr int radixSortMaxWarps = radixSortBlockSize / 32;

//! \brief Trait telling if the type is a specialization of `std::less`.
template< typename T >
struct is_std_less : std::false_type
{};

template< typename T >
struct is_std_less< std::less< T > > : std::true_type
{};

//! \brief Shortcut for \ref is_std_less<T>::value.
template< typename T >
inline constexpr bool is_std_less_v = is_std_less< T >::value;

//! \brief Trait telling if the type is a specialization of `std::greater`.
template< typename T >
struct is_std_greater : std::false_type
{};

template< typename T >
struct is_std_greater< std::greater< T > > : std::true_type
{};

//! \brief Shortcut for \ref is_std_greater<T>::value.
template< typename T >
inline constexpr bool is_std_greater_v = is_std_greater< T >::value;

/**
 * \brief Type of the radix sort key for the given value type.
 *
 * The key is an unsigned integer wide enough to hold all bytes of the value.
 */
template< typename Value >
using RadixSortKey = std::conditional_t< ( sizeof( Value ) > 4 ), std::uint64_t, std::uint32_t >;

/**
 * \brief Maps a value to an order-preserving unsigned sort key.
 *
 * The mapping has the property that for any values `a < b` the keys satisfy
 * `key( a ) < key( b )`:
 *
 * - unsigned values are used as they are,
 * - signed values have the sign bit flipped,
 *   which maps negative values below non-negative values,
 * - floating-point values are bit-cast to an unsigned key and the bits of
 *   negative values are inverted while non-negative values get the sign bit
 *   set, which orders negative values (including infinities) below positive
 *   values in the correct order.
 *
 * The function is `__cuda_callable__` so that it can be used in device code.
 */
template< typename Value >
__cuda_callable__
RadixSortKey< Value >
valueToRadixSortKey( const Value& value )
{
   using Key = RadixSortKey< Value >;
   Key key = 0;
   const auto* bytes = reinterpret_cast< const unsigned char* >( &value );
   unsigned shift = 0;
   for( const unsigned char* byte = bytes; byte < bytes + sizeof( Value ); ++byte ) {
      key |= static_cast< Key >( *byte ) << shift;
      shift += 8;
   }
   if constexpr( std::is_floating_point_v< Value > ) {
      constexpr Key signBit = Key( 1 ) << ( 8 * sizeof( Value ) - 1 );
      if( key & signBit )
         key = ~key;
      else
         key |= signBit;
   }
   else if constexpr( std::is_signed_v< Value > ) {
      key ^= Key( 1 ) << ( 8 * sizeof( Value ) - 1 );
   }
   return key;
}

/**
 * \brief Returns the radix sort key of the value, inverted for the
 * descending order.
 */
template< typename Value, bool descending >
__cuda_callable__
RadixSortKey< Value >
radixSortKey( const Value& value )
{
   auto key = valueToRadixSortKey( value );
   if constexpr( descending )
      key = ~key;
   return key;
}

/**
 * \brief Extracts the digit of the sort key for the given pass.
 *
 * When `descending` is true, the key is inverted so that the ascending
 * radix sort of the inverted keys produces descending order.
 *
 * \param value is the value to extract the digit from.
 * \param pass is the index of the radix sort pass, from 0 (least significant
 *            digit) to `sizeof( Value ) - 1`.
 */
template< typename Value, bool descending >
__cuda_callable__
unsigned
radixSortDigit( const Value& value, int pass )
{
   const auto key = radixSortKey< Value, descending >( value );
   return static_cast< unsigned >( key >> ( radixSortBits * pass ) ) & ( radixSortBuckets - 1 );
}

#if defined( __CUDACC__ ) || defined( __HIP__ )

//! \brief Type of the warp synchronization mask, following
//! \ref Backend::getWarpFullMask (32-bit on CUDA, 64-bit on HIP).
//!
//! \note The warp match intrinsic used by the binning kernels requires
//! compute capability 7.0 or newer on CUDA and gfx8 or newer on HIP.
using RadixSortWarpMask = decltype( Backend::getWarpFullMask() );

/**
 * \brief Counts the set bits in a warp synchronization mask.
 */
[[nodiscard]] __device__
__forceinline__ int
radixSortPopcount( RadixSortWarpMask mask )
{
   if constexpr( sizeof( RadixSortWarpMask ) == 4 )
      return __popc( mask );
   else
      return __popcll( mask );
}

/**
 * \brief Returns the 1-based index of the lowest set bit in a warp
 * synchronization mask, following `__ffs`.
 */
[[nodiscard]] __device__
__forceinline__ int
radixSortFindFirstSet( RadixSortWarpMask mask )
{
   if constexpr( sizeof( RadixSortWarpMask ) == 4 )
      return __ffs( static_cast< unsigned >( mask ) );
   else
      return __ffsll( mask );
}

#endif

#if defined( __CUDACC__ ) || defined( __HIP__ )

/**
 * \brief Shared-memory union of the rank offsets and the staging tile of
 * \ref radixSortOnesweepKernel.
 *
 * The per-(warp, digit) rank offsets are consumed by the computation of the
 * staged positions of the items into registers before the staging tile is
 * written, so the two arrays can share the same storage, which reduces the
 * shared-memory footprint of the kernel and increases the number of thread
 * blocks resident on a streaming multiprocessor.
 */
template< typename Value >
union RadixSortRankTileUnion
{
   int warpOffsets[ radixSortMaxWarps * radixSortBuckets ];
   Value tile[ radixSortTileSize ];
};

#endif

/**
 * \brief Atomically adds the value to the counter of any index type.
 *
 * CUDA provides 64-bit atomic addition only for `unsigned long long`, so
 * wider index types need a bit reinterpretation (the counters are always
 * non-negative).
 */
template< typename Index >
__device__
__forceinline__ void
radixSortAtomicAdd( Index* address, Index value )
{
   if constexpr( sizeof( Index ) == 8 )
      atomicAdd( reinterpret_cast< unsigned long long* >( address ), static_cast< unsigned long long >( value ) );
   else
      atomicAdd( address, value );
}

/**
 * \brief Kernel computing the global histograms of digit counts for all passes.
 *
 * The digit counts of each pass are independent of the order of the elements,
 * so the histograms of all passes can be computed from the input in a single
 * pass. The single-pass binning kernel (see \ref radixSortOnesweepKernel) then
 * needs only the exclusive sums of these histograms as the global base
 * offsets of the digit runs. Each block processes a cyclic sequence of tiles
 * of the input, with the items of each tile loaded into registers and
 * atomically accumulated into shared per-pass histograms, which are then
 * atomically accumulated into the global histograms.
 *
 * \param input is the view of the array to be sorted.
 * \param counts is the view of the global histograms to be filled, with the
 *               entry for the pass `p` and digit `d` stored at
 *               `counts[ p * radixSortBuckets + d ]`. It must be zeroed.
 */
template< typename Value, typename Index, bool descending >
__global__
void
radixSortTotalHistogramKernel(
   const Containers::ArrayView< Value, Devices::GPU, Index > input,
   Containers::ArrayView< Index, Devices::GPU, Index > counts )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   constexpr int passes = static_cast< int >( sizeof( Value ) );
   constexpr int itemsPerThread = radixSortItemsPerThread;
   constexpr int histogramTileSize = radixSortBlockSize * itemsPerThread;
   __shared__ unsigned int s_counts[ passes ][ radixSortBuckets ];

   for( int i = static_cast< int >( threadIdx.x ); i < passes * radixSortBuckets; i += static_cast< int >( blockDim.x ) )
      s_counts[ i / radixSortBuckets ][ i % radixSortBuckets ] = 0;
   __syncthreads();

   const Index n = input.getSize();
   // computed without rounding up to avoid overflow of the index type when
   // the array size is close to its maximum
   const Index numTiles = n / static_cast< Index >( histogramTileSize ) + static_cast< Index >( n % histogramTileSize != 0 );
   for( Index tile = static_cast< Index >( blockIdx.x ); tile < numTiles; tile += static_cast< Index >( gridDim.x ) ) {
      const Index tileBegin = tile * static_cast< Index >( histogramTileSize );
      const Index tileEnd = tileBegin + TNL::min( n - tileBegin, static_cast< Index >( histogramTileSize ) );
      const Index tileSize = tileEnd - tileBegin;

      // load the items of the tile into registers (to overlap the latency);
      // the tile-local loop counter avoids overflow of the index type
      Value values[ itemsPerThread ]{};
      bool valid[ itemsPerThread ]{};
      #pragma unroll
      for( int j = 0; j < itemsPerThread; j++ ) {
         const Index i = j * static_cast< Index >( radixSortBlockSize ) + static_cast< Index >( threadIdx.x );
         if( ( valid[ j ] = i < tileSize ) )
            values[ j ] = input[ tileBegin + i ];
      }
      #pragma unroll
      for( int j = 0; j < itemsPerThread; j++ ) {
         if( valid[ j ] ) {
            #pragma unroll
            for( int p = 0; p < passes; p++ )
               atomicAdd( &s_counts[ p ][ radixSortDigit< Value, descending >( values[ j ], p ) ], 1u );
         }
      }
   }
   __syncthreads();

   for( int i = static_cast< int >( threadIdx.x ); i < passes * radixSortBuckets; i += static_cast< int >( blockDim.x ) )
      radixSortAtomicAdd( &counts[ i ], static_cast< Index >( s_counts[ i / radixSortBuckets ][ i % radixSortBuckets ] ) );
#endif
}

/**
 * \brief Status flags of the tile counters used by
 * \ref radixSortOnesweepKernel.
 *
 * A tile counter is zero until the block processing the tile publishes the
 * tile's digit count (the aggregate), and it is republished with the
 * inclusive prefix through the tile to shorten the lookback of the
 * subsequent blocks. The two most significant bits hold the status, the
 * lower 30 bits hold the value.
 */
constexpr unsigned int radixSortTileAggregate = 1u << 30;
constexpr unsigned int radixSortTilePrefix = 1u << 31;
constexpr unsigned int radixSortTileValueMask = ( 1u << 30 ) - 1;

// the launch bounds keep the occupancy of the binning kernel from
// silently dropping when a change grows the register count: with 256
// threads per block, at most 64 registers are available per thread when
// 4 blocks should stay resident on a multiprocessor (8-byte values need
// more registers, so only 2 blocks are requested for them); HIP is not
// guarded because the attribute was not verified there
#if defined( __CUDACC__ )
   #define RADIX_SORT_LAUNCH_BOUNDS __launch_bounds__( radixSortBlockSize, sizeof( Value ) <= 4 ? 4 : 2 )
#else
   #define RADIX_SORT_LAUNCH_BOUNDS
#endif

/**
 * \brief Kernel performing one digit-binning pass of the single-pass variant.
 *
 * The tiles are assigned to the blocks via a work counter: an atomic
 * increment assigns the tile indices in increasing order, regardless of the
 * block scheduling. Each warp handles a contiguous span of the tile, split
 * into \ref radixSortItemsPerThread rounds of `warpSize` elements, so that
 * the loads and stores of each round are coalesced and the warp-major order
 * of the rank offsets matches the order of the elements in the tile. The
 * intra-tile rank of an element among the elements with the same digit is
 * computed with the warp match intrinsic, which is O(1) in the tile size
 * instead of a linear scan over the digits. After the ranks are computed,
 * the items are staged into a shared-memory tile grouped by digit, from
 * which they are written out with coalesced global stores. The per-digit
 * base offsets are not loaded from a global histogram, but computed on the
 * fly with a decoupled lookback: after the scan phase, each thread
 * (handling one digit) publishes the tile's aggregate count of its digit
 * and then walks the counters of the preceding tiles, accumulating their
 * aggregates, until it finds a counter holding the inclusive prefix, which
 * together with the accumulated aggregates gives the exclusive offset of
 * the tile.
 * The first tile has no predecessors, so its strip-relative offset is zero.
 * Publishing the aggregate before waiting on any predecessor guarantees
 * that the lookback of all blocks always makes progress, without requiring
 * all blocks of the grid to be co-resident - it relies only on the standard
 * forward-progress assumption of resident thread blocks (the same assumption
 * as in CUB's single-pass radix sort). After the lookback,
 * the block republishes its inclusive prefix, which shortens the lookback
 * of the subsequent blocks.
 *
 * The kernel processes one strip of at most `2^28` elements of the input,
 * which bounds the lookback values to 30 bits. The base offsets of the
 * digit runs, accumulated over the preceding strips, are passed in
 * \p digitBase and added to the strip-relative offsets computed by the
 * lookback. The block processing the last tile of the strip publishes the
 * strip's inclusive prefix in its tile counter, from which the driver
 * advances \p digitBase for the next strip.
 *
 * \param input is the view of the strip of the array to be sorted.
 * \param output is the view of the auxiliary array for the scattered
 *               elements (indexed globally, i.e. it spans the whole array).
 * \param digitBase is the view of the base offsets of the digit runs, with
 *                  the entry for the pass `p` and digit `d` stored at
 *                  `digitBase[ p * radixSortBuckets + d ]`.
 * \param tileStatus is the view of the tile status counters, with one counter
 *                   per (tile, digit) plus one work counter at the end. It
 *                   must be zeroed.
 * \param numTiles is the number of tiles of the strip.
 * \param pass is the index of the radix sort pass.
 */
template< typename Value, typename Index, bool descending >
__global__
void RADIX_SORT_LAUNCH_BOUNDS
radixSortOnesweepKernel(
   const Containers::ArrayView< Value, Devices::GPU, Index > input,
   Containers::ArrayView< Value, Devices::GPU, Index > output,
   const Containers::ArrayView< Index, Devices::GPU, Index > digitBase,
   Containers::ArrayView< unsigned int, Devices::GPU, Index > tileStatus,
   Index numTiles,
   int pass )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ Index s_base[ radixSortBuckets ];
   __shared__ int s_digitStart[ radixSortBuckets + 1 ];
   __shared__ int s_singleDigit;
   __shared__ Index s_claimedTile;
   __shared__ RadixSortRankTileUnion< Value > s_rankTile;

   // the launch uses blockSize.x == radixSortBuckets, so each thread handles
   // exactly one digit in the per-digit phases below
   const int digit = static_cast< int >( threadIdx.x );
   s_digitStart[ 0 ] = 0;

   const Index n = input.getSize();
   unsigned int* const workCounter = &tileStatus[ numTiles * radixSortBuckets ];

   const int warpId = static_cast< int >( threadIdx.x ) / warpSize;
   const int lane = static_cast< int >( threadIdx.x ) % warpSize;
   constexpr int itemsPerThread = radixSortItemsPerThread;
   const RadixSortWarpMask fullMask = Backend::getWarpFullMask();
   const RadixSortWarpMask laneMaskLt =
      ( static_cast< RadixSortWarpMask >( 1 ) << lane ) - static_cast< RadixSortWarpMask >( 1 );

   // the work counter assigns tile indices in increasing order, so the block
   // processing a tile can rely on all its predecessors being claimed
   while( true ) {
      if( threadIdx.x == 0 )
         s_claimedTile = static_cast< Index >( atomicAdd( workCounter, 1u ) );
      __syncthreads();
      const Index tile = s_claimedTile;
      if( tile >= numTiles )
         break;
      const Index tileBegin = tile * static_cast< Index >( radixSortTileSize );
      const Index tileEnd = tileBegin + TNL::min( n - tileBegin, static_cast< Index >( radixSortTileSize ) );
      const int tileSize = static_cast< int >( tileEnd - tileBegin );

      // zero the per-(warp, digit) offsets of this tile
      for( int i = digit; i < radixSortMaxWarps * radixSortBuckets; i += static_cast< int >( blockDim.x ) )
         s_rankTile.warpOffsets[ i ] = 0;
      __syncthreads();

      // load the items of all rounds (to overlap the memory latency);
      // each warp handles a contiguous span of the tile, split into
      // consecutive rounds of warpSize elements, so that the loads of each
      // round are coalesced and the warp-major order of the rank offsets
      // below matches the order of the elements in the tile
      Value values[ itemsPerThread ]{};
      unsigned char digits[ itemsPerThread ]{};
      #pragma unroll
      for( int j = 0; j < itemsPerThread; j++ ) {
         const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
         if( idxInTile < tileSize ) {
            values[ j ] = input[ tileBegin + idxInTile ];
            digits[ j ] = static_cast< unsigned char >( radixSortDigit< Value, descending >( values[ j ], pass ) );
         }
      }

      // rank the items: the lowest lane of each group of equal digits
      // atomically adds the group size to the running (warp, digit) offset
      // and broadcasts the returned value within the group, which gives the
      // offset of the item's round within the warp
      int itemOffsets[ itemsPerThread ]{};
      #pragma unroll
      for( int j = 0; j < itemsPerThread; j++ ) {
         // active lanes of the warp form a prefix of the warp, so the mask
         // of lanes participating in the warp match can be computed directly
         const int activeInWarp = TNL::max(
            0,
            TNL::min(
               static_cast< int >( warpSize ),
               tileSize - warpId * ( warpSize * itemsPerThread ) - j * static_cast< int >( warpSize ) ) );
         const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
         if( activeInWarp > 0 && idxInTile < tileSize ) {
            const RadixSortWarpMask activeMask =
               activeInWarp >= static_cast< int >( warpSize )
                  ? fullMask
                  : ( static_cast< RadixSortWarpMask >( 1 ) << activeInWarp ) - static_cast< RadixSortWarpMask >( 1 );
            const RadixSortWarpMask match = __match_any_sync( activeMask, static_cast< int >( digits[ j ] ) );
            const int rank = radixSortPopcount( match & laneMaskLt );
            const int leaderLane = radixSortFindFirstSet( match ) - 1;
            int roundOffset = 0;
            if( lane == leaderLane )
               roundOffset =
                  atomicAdd( &s_rankTile.warpOffsets[ warpId * radixSortBuckets + digits[ j ] ], radixSortPopcount( match ) );
            roundOffset = __shfl_sync( activeMask, roundOffset, leaderLane );
            itemOffsets[ j ] = roundOffset + rank;
         }
      }
      __syncthreads();

      // exclusive scan of the warp offsets of each digit (in place); the
      // total count of the thread's digit stays in a register for the
      // lookback below
      int digitCount = 0;
      #pragma unroll
      for( int w = 0; w < radixSortMaxWarps; w++ ) {
         const int idx = w * radixSortBuckets + digit;
         const int count = s_rankTile.warpOffsets[ idx ];
         s_rankTile.warpOffsets[ idx ] = digitCount;
         digitCount += count;
      }
      s_digitStart[ digit + 1 ] = digitCount;
      __syncthreads();

      // publish the tile's aggregate count of this thread's digit before
      // waiting on any predecessor, which guarantees that the lookback of
      // all blocks always makes progress
      atomicExch( &tileStatus[ tile * radixSortBuckets + digit ], radixSortTileAggregate | digitCount );

      // turn the digit counts into the tile-local starts of the digit runs
      // by a raking scan: each thread of the first warp reads a segment of
      // counts, the segment totals are scanned across the warp, and each
      // thread writes the scanned starts of its segment; this has about the
      // same instruction count as a serial scan by one thread, but the
      // latency of only a few dozen cycles
      if( warpId == 0 ) {
         constexpr int digitsPerThread = radixSortBuckets / Backend::getWarpSize();
         int counts[ digitsPerThread ];
         int sum = 0;
         #pragma unroll
         for( int k = 0; k < digitsPerThread; k++ ) {
            counts[ k ] = s_digitStart[ lane * digitsPerThread + k + 1 ];
            sum += counts[ k ];
         }
         int inclusive = sum;
         #pragma unroll
         for( int off = 1; off < warpSize; off <<= 1 ) {
            const int v = __shfl_up_sync( fullMask, inclusive, off );
            if( lane >= off )
               inclusive += v;
         }
         int running = inclusive - sum;
         #pragma unroll
         for( int k = 0; k < digitsPerThread; k++ ) {
            running += counts[ k ];
            s_digitStart[ lane * digitsPerThread + k + 1 ] = running;
         }

         // if all items of the tile fall into the same digit bin, the bin is
         // recorded so that the tile can be written out without staging; at
         // most one bin can own all items, since the counts sum to the tile
         // size, so the write below has a single writer
         int single = -1;
         #pragma unroll
         for( int k = 0; k < digitsPerThread; k++ )
            if( counts[ k ] == tileSize )
               single = lane * digitsPerThread + k;
         const RadixSortWarpMask ballot = __ballot_sync( fullMask, single >= 0 );
         if( single >= 0 )
            s_singleDigit = single;
         else if( lane == 0 && ballot == 0 )
            s_singleDigit = -1;
      }
      __syncthreads();

      // fold the digit start and the warp offset of each item into its rank
      // offset, which turns it into the tile-relative staged position; this
      // is the last use of the rank offsets, so the staging tile written
      // below can alias them in shared memory (see RadixSortRankTileUnion)
      if( s_singleDigit < 0 ) {
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ ) {
            const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
            if( idxInTile < tileSize )
               itemOffsets[ j ] +=
                  s_digitStart[ digits[ j ] ] + s_rankTile.warpOffsets[ warpId * radixSortBuckets + digits[ j ] ];
         }
      }
      // no thread may read the rank offsets after this barrier, so the
      // staging tile may clobber them
      __syncthreads();

      // stage the items into the shared tile, grouped by digit
      if( s_singleDigit < 0 ) {
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ ) {
            const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
            if( idxInTile < tileSize )
               s_rankTile.tile[ itemOffsets[ j ] ] = values[ j ];
         }
      }
      __syncthreads();

      // decoupled lookback: accumulate the aggregates of the preceding tiles
      // until an inclusive prefix is found; the first tile has no
      // predecessors, so its strip-relative offset is zero
      Index stripExclusive = 0;
      if( tile != 0 ) {
         unsigned int accumulated = 0;
         bool foundPrefix = false;
         for( Index predecessor = tile; predecessor-- > 0; ) {
            unsigned int word;
            // spin until the predecessor publishes its aggregate or prefix;
            // the status and the value share one word, so no acquire-release
            // ordering is needed - the volatile load bypasses L1 and the
            // atomic exchange is performed in L2, which makes the word
            // visible in bounded time
            while( ( word = *reinterpret_cast< const volatile unsigned int* >(
                        &tileStatus[ predecessor * radixSortBuckets + digit ] ) )
                   < radixSortTileAggregate )
            {
            }
            if( word & radixSortTilePrefix )
               foundPrefix = true;
            accumulated += word & radixSortTileValueMask;
            if( foundPrefix )
               break;
         }
         // no prefix was found: the aggregates of all predecessors were
         // accumulated, which is the strip-relative exclusive offset
         stripExclusive = static_cast< Index >( accumulated );
      }
      s_base[ digit ] = digitBase[ static_cast< Index >( pass ) * radixSortBuckets + digit ] + stripExclusive;
      // republish with the strip-relative inclusive prefix through this
      // tile, which shortens the lookback of the subsequent blocks; the
      // block of the last tile of the strip publishes the strip's total
      // count of its digit in this counter, from which the driver advances
      // the base offsets of the digit runs for the next strip
      atomicExch(
         &tileStatus[ tile * radixSortBuckets + digit ],
         radixSortTilePrefix | static_cast< unsigned int >( stripExclusive + digitCount ) );
      __syncthreads();

      if( s_singleDigit >= 0 ) {
         // all items of the tile fall into the same digit bin: write them
         // out directly from registers, skipping the staged tile; the
         // intra-tile rank of an item is its index in the tile, so the
         // stores are stable and consecutive
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ ) {
            const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
            if( idxInTile < tileSize )
               output[ s_base[ s_singleDigit ] + idxInTile ] = values[ j ];
         }
      }
      else {
         // write the tile out with coalesced stores: consecutive positions of
         // the staged tile belong to the same digit run (or to adjacent runs),
         // so the global stores are consecutive as well; the digit owning each
         // position is found by a binary search over the digit starts
         for( int i = digit; i < tileSize; i += static_cast< int >( blockDim.x ) ) {
            int lo = 0;
            int hi = radixSortBuckets;
            while( lo + 1 < hi ) {
               const int mid = ( lo + hi ) / 2;
               if( s_digitStart[ mid ] <= i )
                  lo = mid;
               else
                  hi = mid;
            }
            output[ s_base[ lo ] + ( i - s_digitStart[ lo ] ) ] = s_rankTile.tile[ i ];
         }
      }
      __syncthreads();
   }
#endif
}

#undef RADIX_SORT_LAUNCH_BOUNDS

/**
 * \brief Kernel advancing the base offsets of the digit runs after a strip.
 *
 * The last tile of the preceding strip has published its inclusive prefix
 * in its tile counter (see \ref radixSortOnesweepKernel), which is the
 * strip's total count of each digit. The totals are added to the base
 * offsets of the given pass, so that the following strip scatters its
 * elements after the elements of the preceding strips.
 *
 * \param digitBase is the view of the base offsets of the digit runs, with
 *                  the entry for the pass `p` and digit `d` stored at
 *                  `digitBase[ p * radixSortBuckets + d ]`.
 * \param tileStatus is the view of the tile status counters of the preceding
 *                   strip (see \ref radixSortOnesweepKernel).
 * \param numTiles is the number of tiles of the preceding strip.
 * \param pass is the index of the radix sort pass.
 */
template< typename Index >
__global__
void
radixSortAdvanceBaseKernel(
   Containers::ArrayView< Index, Devices::GPU, Index > digitBase,
   const Containers::ArrayView< unsigned int, Devices::GPU, Index > tileStatus,
   Index numTiles,
   int pass )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index digit = static_cast< Index >( threadIdx.x );
   const unsigned int word = tileStatus[ ( numTiles - 1 ) * radixSortBuckets + digit ];
   digitBase[ static_cast< Index >( pass ) * radixSortBuckets + digit ] += word & radixSortTileValueMask;
#endif
}

/**
 * \brief Sorts the given view using LSD radix sort.
 *
 * The algorithm processes the bytes of the order-preserving sort key
 * (see \ref valueToRadixSortKey) from the least significant to the most
 * significant one. All passes are stable, so the overall sort is stable.
 *
 * The single-pass variant is used: the digit histograms of all passes are
 * computed from the input in one upfront pass (the digit counts do not
 * depend on the order of the elements, which changes between the passes),
 * and each pass then consists of a single binning kernel which exchanges
 * the per-(tile, digit) aggregates and inclusive prefixes with the
 * preceding blocks via decoupled lookback (see \ref radixSortOnesweepKernel),
 * so that each element is read and written only once per pass.
 *
 * The input is processed in strips of at most `2^28` elements, which bounds
 * the lookback values to 30 bits; the base offsets of the digit runs are
 * advanced between the strips (see \ref radixSortAdvanceBaseKernel).
 *
 * A pass whose histogram consists of a single non-zero bucket maps all
 * elements to the same digit, so its stable partition is the identity
 * permutation: such passes are detected from the upfront histogram on the
 * host and their sweeps are skipped in all strips. The digit counts are
 * permutation-invariant, so the histograms of the later passes stay valid
 * regardless of which earlier passes are skipped.
 *
 * \param view is the view of the array to be sorted in place.
 * \tparam descending tells whether to sort in descending order.
 */
template< typename Value, typename Device, typename Index, bool descending >
void
radixSort( Containers::ArrayView< Value, Device, Index > view )
{
   static_assert( std::is_same_v< Device, Devices::GPU >, "RadixSort is implemented only for the GPU device" );
   static_assert( std::is_arithmetic_v< Value >, "RadixSort supports only arithmetic value types" );
   static_assert( sizeof( Value ) <= 8, "RadixSort supports value types with at most 8 bytes" );

#if defined( __CUDACC__ ) || defined( __HIP__ )
   // early exit for empty or single-element arrays (already sorted)
   if( view.getSize() <= 1 )
      return;

   const Index n = view.getSize();
   constexpr int passes = static_cast< int >( sizeof( Value ) );

   // computed without rounding up to avoid overflow of n + radixSortTileSize
   // when the index type is signed and n is close to its maximum
   const Index numTiles = n / static_cast< Index >( radixSortTileSize ) + static_cast< Index >( n % radixSortTileSize != 0 );
   const int multiprocessors = Backend::getDeviceMultiprocessors( Backend::getDevice() );
   const Index maxBlocks = static_cast< Index >( multiprocessors ) * 32;

   Containers::Array< Value, Device, Index > aux( n );

   auto input = view;
   auto output = aux.getView();

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = radixSortBlockSize;
   launch_config.gridSize.x = static_cast< unsigned >( TNL::min( numTiles, maxBlocks ) );

   // the digit counts are independent of the order of the elements, which
   // changes between the passes, so the histograms of all passes are
   // computed upfront; the exclusive scan of each histogram then gives the
   // global base offsets of the digit runs
   Containers::Array< Index, Device, Index > digitBase( passes * radixSortBuckets );
   digitBase.setValue( 0 );

   constexpr auto totalHistogramKernel = radixSortTotalHistogramKernel< Value, Index, descending >;
   Backend::launchKernelAsync( totalHistogramKernel, launch_config, input, digitBase.getView() );

   // copy the digit counts to the host (the synchronous copy also orders the
   // rest of the computation after the histogram kernel): the exclusive
   // scans of the 256-counter histograms are cheaper on the host than in
   // per-pass kernel launches with host synchronization, and the raw counts
   // additionally allow the skippable passes to be detected - a pass whose
   // histogram has a single non-zero bucket maps all elements to the same
   // digit, so its stable partition is the identity permutation and its
   // sweeps can be skipped in all strips
   Index hostCounts[ passes * radixSortBuckets ];
   Backend::memcpy( hostCounts, digitBase.getData(), sizeof( hostCounts ), Backend::MemcpyDeviceToHost );
   unsigned int skippedPasses = 0;
   for( int p = 0; p < passes; p++ ) {
      int nonZeroBuckets = 0;
      Index total = 0;
      for( int d = 0; d < radixSortBuckets; d++ ) {
         const Index count = hostCounts[ p * radixSortBuckets + d ];
         nonZeroBuckets += count != 0;
         hostCounts[ p * radixSortBuckets + d ] = total;
         total += count;
      }
      if( nonZeroBuckets == 1 )
         skippedPasses |= 1u << p;
   }

   // upload the base offsets of the digit runs (the entries of the skipped
   // passes are only ever consumed by their own sweeps, so their content
   // does not matter)
   Backend::memcpy( digitBase.getData(), hostCounts, sizeof( hostCounts ), Backend::MemcpyHostToDevice );

   // the input is processed in strips of up to 2^28 elements: the values
   // exchanged by the decoupled lookback must fit into 30 bits, and the
   // tile status array (half a byte per element, one counter per tile and
   // digit) stays reasonably small
   constexpr Index onesweepStripSize = ( static_cast< Index >( 1 ) << 28 );
   const Index numStrips = n / onesweepStripSize + static_cast< Index >( n % onesweepStripSize != 0 );
   const Index maxStripTiles = TNL::min( numTiles, onesweepStripSize / radixSortTileSize );

   // one status counter per (tile, digit), plus the work counter
   Containers::Array< unsigned int, Device, Index > tileStatus( maxStripTiles * radixSortBuckets + 1 );

   constexpr auto onesweepKernel = radixSortOnesweepKernel< Value, Index, descending >;
   constexpr auto advanceBaseKernel = radixSortAdvanceBaseKernel< Index >;

   // all kernels are launched on the same stream, so their execution is
   // ordered by the stream
   int executedPasses = 0;
   for( int pass = 0; pass < passes; pass++ ) {
      // the sweeps of a skipped pass are the identity permutation, so they
      // are omitted in every strip, along with the advance of the pass's
      // base offsets (which only its own sweeps would consume)
      if( skippedPasses & ( 1u << pass ) )
         continue;
      for( Index strip = 0; strip < numStrips; strip++ ) {
         const Index stripBegin = strip * onesweepStripSize;
         const Index stripEnd = TNL::min( n, stripBegin + onesweepStripSize );
         const Index stripSize = stripEnd - stripBegin;
         // computed without rounding up to avoid overflow of the index type
         // when the strip size is close to its maximum
         const Index stripNumTiles = stripSize / radixSortTileSize + static_cast< Index >( stripSize % radixSortTileSize != 0 );

         // zeroing the status also resets the work counter
         tileStatus.setValue( 0 );
         launch_config.gridSize.x = static_cast< unsigned >( TNL::min( stripNumTiles, maxBlocks ) );
         Backend::launchKernelAsync(
            onesweepKernel,
            launch_config,
            input.getView( stripBegin, stripEnd ),
            output,
            digitBase.getView(),
            tileStatus.getView(),
            stripNumTiles,
            pass );

         // the last tile of the strip has published the strip's digit totals
         // in its tile counters, which advance the base offsets for the next
         // strip
         if( strip + 1 < numStrips ) {
            launch_config.gridSize.x = 1;
            Backend::launchKernelAsync(
               advanceBaseKernel, launch_config, digitBase.getView(), tileStatus.getView(), stripNumTiles, pass );
         }
      }
      input.swap( output );
      executedPasses++;
   }
   Backend::streamSynchronize( launch_config.stream );

   // for an odd number of executed passes the sorted data ends up in the
   // auxiliary array; zero executed passes means that all elements share
   // every digit, i.e. the input is already sorted in place
   if( executedPasses % 2 == 1 )
      view = input;
#else
   throw Exceptions::NotImplementedError( "RadixSort is supported only when CUDA or HIP is enabled." );
#endif
}

}  // namespace TNL::Algorithms::Sorting::detail
