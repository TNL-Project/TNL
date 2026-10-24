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
#include <TNL/Algorithms/detail/GPUScan/GPUBlockScan.h>
#include <TNL/Algorithms/fill.h>

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

//! \brief Array size threshold below which the tile status regions of all
//! passes are zeroed by a single upfront fill.
//!
//! The aggregated fill replaces one fill launch per pass, but it pays off
//! only while the sweeps' streaming traffic between the fill and the last
//! pass stays small enough for the zeroed regions to stay cache-hot (at
//! about 2^22 elements on the current GPUs); larger sorts share the first
//! region and zero it right before each sweep (see \ref radixSort).
constexpr int radixSortAggregatedFillLimit = 1 << 22;

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
      const int tileSize = static_cast< int >( tileEnd - tileBegin );

      // load the items of the tile into registers (to overlap the latency);
      // the loads are addressed with 32-bit tile-relative indices from the
      // base pointer of the tile, which avoids the 64-bit address chain
      Value values[ itemsPerThread ]{};
      bool valid[ itemsPerThread ]{};
      const Value* const tileInput = input.getData() + tileBegin;
      #pragma unroll
      for( int j = 0; j < itemsPerThread; j++ ) {
         const int i = j * radixSortBlockSize + static_cast< int >( threadIdx.x );
         if( ( valid[ j ] = i < tileSize ) )
            values[ j ] = tileInput[ i ];
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
 * \brief Kernel scanning the digit histograms and deriving the skippable
 * passes and the sweep buffers from them.
 *
 * The global base offsets of the digit runs are the exclusive scans of the
 * per-pass histograms; the kernel computes the scans of all (at most 8)
 * passes, replacing the per-pass scan launches of the driver. The launch
 * uses a single block of \ref radixSortBuckets threads, so the raw counters
 * of one pass are held one per thread and the block-wide exclusive scan
 * written back in place scans one pass per iteration of the main loop.
 *
 * A pass whose histogram consists of a single non-zero bucket maps all
 * elements to the same digit, so the stable partition of its sweep is the
 * identity permutation and the sweep can be skipped. The flags are computed
 * on the device so that the driver does not have to synchronize the host to
 * learn them: the host launches the sweeps of all passes and each sweep
 * adjusts itself (see \ref radixSortOnesweepKernel). The non-zero buckets
 * of each pass are counted with warp ballots, which predicate on the raw
 * counters held in registers; thread 0 then computes the skip flag of
 * each pass, the running parity of the executed passes before each pass,
 * and the parity of the total number of executed passes, which is a short
 * sequential prefix.
 *
 * \param counts is the view of the digit histograms filled by
 *               \ref radixSortTotalHistogramKernel, with the entry for the
 *               pass `p` and digit `d` stored at
 *               `counts[ p * radixSortBuckets + d ]`; the raw counts are
 *               replaced by their exclusive scans in place.
 * \param flags is the view of `2 * passes + 1` integers to be written:
 *              `flags[ p ]` tells whether the pass `p` is skipped,
 *              `flags[ passes + p ]` is the parity of the number of passes
 *              executed before the pass `p`, and `flags[ 2 * passes ]` is
 *              the parity of the total number of executed passes.
 */
template< typename Value, typename Index >
__global__
void
radixSortProcessHistogramsKernel(
   Containers::ArrayView< Index, Devices::GPU, Index > counts,
   Containers::ArrayView< int, Devices::GPU, Index > flags )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   constexpr int passes = static_cast< int >( sizeof( Value ) );
   // the launch uses blockDim.x == radixSortBuckets, so the block holding
   // one counter per thread scans the 256 counters of one pass
   using BlockScan =
      Algorithms::detail::GPUBlockScan< Algorithms::detail::ScanType::Exclusive, radixSortBuckets, std::plus<>, Index >;
   __shared__ typename BlockScan::Storage s_scanStorage;
   __shared__ int s_nonZeroBuckets[ passes ];
   if( static_cast< int >( threadIdx.x ) < passes )
      s_nonZeroBuckets[ threadIdx.x ] = 0;
   __syncthreads();

   const int digit = static_cast< int >( threadIdx.x );
   const int lane = static_cast< int >( threadIdx.x ) % warpSize;
   const RadixSortWarpMask fullMask = Backend::getWarpFullMask();
   for( int p = 0; p < passes; p++ ) {
      // the block-wide exclusive scan of the pass's raw counters is written
      // back in place, giving the global base offsets of the digit runs
      const Index count = counts[ static_cast< Index >( p ) * radixSortBuckets + digit ];
      const Index base = BlockScan::scan( std::plus<>{}, Index( 0 ), count, static_cast< int >( threadIdx.x ), s_scanStorage );
      counts[ static_cast< Index >( p ) * radixSortBuckets + digit ] = base;

      // count the non-zero buckets of the pass: a single non-zero bucket
      // means the pass maps all elements to the same digit and its sweep
      // can be skipped
      const RadixSortWarpMask ballot = __ballot_sync( fullMask, count != 0 );
      if( lane == 0 && ballot != 0 )
         atomicAdd( &s_nonZeroBuckets[ p ], radixSortPopcount( ballot ) );
   }
   __syncthreads();

   // the parity prefix is sequential over the (at most 8) passes, so it is
   // computed by a single thread
   if( threadIdx.x == 0 ) {
      int parity = 0;
      #pragma unroll
      for( int p = 0; p < passes; p++ ) {
         const int skip = static_cast< int >( s_nonZeroBuckets[ p ] == 1 );
         flags[ p ] = skip;
         flags[ passes + p ] = parity;
         parity ^= 1 - skip;
      }
      flags[ 2 * passes ] = parity;
   }
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
 * The sweeps of the executed passes alternate between the input array
 * \p arrA and the auxiliary array \p arrB. The host does not know which
 * passes are skipped (see \ref radixSortProcessHistogramsKernel), so it
 * launches the sweeps of all passes and each sweep selects its source and
 * target buffer itself: a skipped pass exits immediately, an executed pass
 * `p` reads the strip of \p arrB when the parity of the passes executed
 * before it is odd (of \p arrA otherwise) and writes the scattered elements
 * to the other buffer.
 *
 * \param arrA is the view of the whole array being sorted; it is one of the
 *             two ping-pong buffers.
 * \param arrB is the view of the whole auxiliary array; it is the other
 *             ping-pong buffer.
 * \param digitBase is the view of the base offsets of the digit runs, with
 *                  the entry for the pass `p` and digit `d` stored at
 *                  `digitBase[ p * radixSortBuckets + d ]`.
 * \param tileStatus is the view of the tile status counters, with one counter
 *                   per (tile, digit) plus one work counter at the end. It
 *                   must be zeroed.
 * \param flags are the pass flags computed by \ref radixSortProcessHistogramsKernel:
 *              `flags[ pass ]` tells whether this pass is skipped and
 *              `flags[ passes + pass ]` tells whether this pass reads the
 *              strip of \p arrB (otherwise the strip of \p arrA) and writes
 *              to the other buffer.
 * \param stripBegin is the index of the first element of the strip in the
 *                   buffers.
 * \param stripEnd is the index one past the last element of the strip.
 * \param numTiles is the number of tiles of the strip.
 * \param pass is the index of the radix sort pass.
 */
template< typename Value, typename Index, bool descending >
__global__
void RADIX_SORT_LAUNCH_BOUNDS
radixSortOnesweepKernel(
   Containers::ArrayView< Value, Devices::GPU, Index > arrA,
   Containers::ArrayView< Value, Devices::GPU, Index > arrB,
   const Containers::ArrayView< Index, Devices::GPU, Index > digitBase,
   Containers::ArrayView< unsigned int, Devices::GPU, Index > tileStatus,
   const Containers::ArrayView< const int, Devices::GPU, Index > flags,
   Index stripBegin,
   Index stripEnd,
   Index numTiles,
   int pass )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   constexpr int passes = static_cast< int >( sizeof( Value ) );

   // a skipped pass maps all elements to the same digit, so its sweep is the
   // identity permutation and is skipped; the tile status counters stay
   // zeroed, so the following advance of the base offsets is a no-op
   if( flags[ pass ] )
      return;

   // the host launches the sweeps of all passes without knowing which of
   // them are skipped: the sweep selects its source and target buffer from
   // the parity of the passes executed before it
   const bool srcIsB = flags[ passes + pass ] != 0;
   const auto input = ( srcIsB ? arrB : arrA ).getView( stripBegin, stripEnd );
   auto output = srcIsB ? arrA : arrB;

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

      // the loads and stores of the tile are addressed with 32-bit
      // tile-relative indices from the base pointer of the tile
      const Value* const tileInput = input.getData() + tileBegin;

      // zero the per-(warp, digit) offsets of this tile
      for( int i = digit; i < radixSortMaxWarps * radixSortBuckets; i += static_cast< int >( blockDim.x ) )
         s_rankTile.warpOffsets[ i ] = 0;
      __syncthreads();

      // the entries of values and digits of the rounds beyond the tail of
      // a partial tile stay uninitialized; the guarded loops below never
      // read them (their lanes do not participate in the warp matches);
      // each warp handles a contiguous span of the tile, split into
      // consecutive rounds of warpSize elements, so that the loads of each
      // round are coalesced and the warp-major order of the rank offsets
      // below matches the order of the elements in the tile
      Value values[ itemsPerThread ];
      unsigned char digits[ itemsPerThread ];
      #pragma unroll
      for( int j = 0; j < itemsPerThread; j++ ) {
         const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
         if( idxInTile < tileSize ) {
            values[ j ] = tileInput[ idxInTile ];
            digits[ j ] = static_cast< unsigned char >( radixSortDigit< Value, descending >( values[ j ], pass ) );
         }
      }

      // rank the items: every lane reads the running (warp, digit) offset of
      // its digit with a plain shared load and the lowest lane of each group
      // of equal digits (the lane with no lower-lane peer) stores the offset
      // advanced by the group size, which gives the offset of the item's
      // round within the warp; the warp synchronizations order the reads of
      // a round before its stores, so the lanes of a group share the same
      // base and no atomics are needed - each round touches every
      // (warp, digit) counter with a single store
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
            volatile int* counter = &s_rankTile.warpOffsets[ warpId * radixSortBuckets + digits[ j ] ];
            const int roundOffset = *counter;
            __syncwarp( activeMask );
            if( rank == 0 )
               *counter = roundOffset + radixSortPopcount( match );
            __syncwarp( activeMask );
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
      // shift the base of the digit's run by the tile-local start of the
      // run, so that the staged writeout stores with a single shared operand
      // (the start of the single-digit run is zero, so the direct writeout
      // path below is unaffected by the shift)
      s_base[ digit ] = digitBase[ static_cast< Index >( pass ) * radixSortBuckets + digit ] + stripExclusive
                      - static_cast< Index >( s_digitStart[ digit ] );
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
         // position is recomputed from the staged value instead of searching
         // the digit starts - it is a pure function of the value and the pass
         for( int i = digit; i < tileSize; i += static_cast< int >( blockDim.x ) ) {
            const int dig = radixSortDigit< Value, descending >( s_rankTile.tile[ i ], pass );
            output[ s_base[ dig ] + i ] = s_rankTile.tile[ i ];
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
 * For a skipped pass the tile counters are still zero, so the advance adds
 * zeros to the (never used) base offsets of the pass and the kernel is a
 * no-op.
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
 * \brief Kernel copying the sorted data back from the auxiliary array.
 *
 * The sweeps of the executed passes alternate between the array being
 * sorted and the auxiliary array, so the sorted data ends up in the
 * auxiliary array when the number of executed passes is odd. The driver
 * does not learn the parity on the host, so the copy is launched
 * unconditionally and the kernel exits immediately when the parity flag is
 * even - including the case of zero executed passes, i.e. when the input is
 * already sorted.
 *
 * \param view is the view of the whole array being sorted (the destination).
 * \param aux is the view of the whole auxiliary array (the source).
 * \param flags are the pass flags computed by \ref radixSortProcessHistogramsKernel;
 *              the copy is performed only when `flags[ 2 * passes ]` is odd.
 */
template< typename Value, typename Index >
__global__
void
radixSortCopyBackKernel(
   Containers::ArrayView< Value, Devices::GPU, Index > view,
   const Containers::ArrayView< Value, Devices::GPU, Index > aux,
   const Containers::ArrayView< const int, Devices::GPU, Index > flags )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   constexpr int passes = static_cast< int >( sizeof( Value ) );
   if( flags[ 2 * passes ] == 0 )
      return;
   const Index n = view.getSize();
   const Index stride = static_cast< Index >( gridDim.x ) * blockDim.x;
   const Index first = static_cast< Index >( blockIdx.x ) * blockDim.x + threadIdx.x;
   if( first >= n )
      return;
   // computed without accumulation to avoid overflow of the index type when
   // n is close to its maximum (first + steps * stride can exceed it)
   const Index steps = ( n - first ) / stride + static_cast< Index >( ( n - first ) % stride != 0 );
   for( Index k = 0; k < steps; k++ )
      view[ first + k * stride ] = aux[ first + k * stride ];
#endif
}

/**
 * \brief Kernel sorting a single tile of at most \ref radixSortTileSize
 * elements in shared memory.
 *
 * Arrays of at most one tile do not benefit from the multi-tile machinery
 * of the general driver: the histograms, the decoupled lookback and the
 * ping-pong buffers are pure overhead at this size. A single thread block
 * loads the whole array into shared memory, performs all digit-binning
 * passes there, and writes the sorted data back, so the sort costs a single
 * kernel launch and needs no temporary buffers at all. The stable ranking
 * of the items is identical to \ref radixSortOnesweepKernel (warp match +
 * raking scan), and the permutations of the passes ping-pong between two
 * shared-memory tiles. A pass mapping all elements to the same digit is the
 * identity permutation, so its permute is skipped and the tiles are not
 * swapped.
 *
 * \param view is the view of the array to be sorted in place; its size must
 *              be at most \ref radixSortTileSize.
 */
template< typename Value, typename Index, bool descending >
__global__
void
radixSortSingleTileKernel( Containers::ArrayView< Value, Devices::GPU, Index > view )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   constexpr int passes = static_cast< int >( sizeof( Value ) );
   __shared__ Value s_tiles[ 2 ][ radixSortTileSize ];
   __shared__ int s_warpOffsets[ radixSortMaxWarps * radixSortBuckets ];
   __shared__ int s_digitStart[ radixSortBuckets + 1 ];
   __shared__ int s_singleDigit;

   // the whole array fits into one tile, so the tile size is the array size
   const int n = static_cast< int >( view.getSize() );
   const int digit = static_cast< int >( threadIdx.x );
   const int warpId = static_cast< int >( threadIdx.x ) / warpSize;
   const int lane = static_cast< int >( threadIdx.x ) % warpSize;
   constexpr int itemsPerThread = radixSortItemsPerThread;
   const RadixSortWarpMask fullMask = Backend::getWarpFullMask();
   const RadixSortWarpMask laneMaskLt =
      ( static_cast< RadixSortWarpMask >( 1 ) << lane ) - static_cast< RadixSortWarpMask >( 1 );

   // load the input into the first shared tile with coalesced loads
   for( int i = digit; i < n; i += static_cast< int >( blockDim.x ) )
      s_tiles[ 0 ][ i ] = view[ i ];
   s_digitStart[ 0 ] = 0;
   __syncthreads();

   // the permutations of the passes ping-pong between the two shared tiles
   int cur = 0;
   for( int pass = 0; pass < passes; pass++ ) {
      // zero the per-(warp, digit) offsets of this pass
      for( int i = digit; i < radixSortMaxWarps * radixSortBuckets; i += static_cast< int >( blockDim.x ) )
         s_warpOffsets[ i ] = 0;
      __syncthreads();

      // load the items of the current tile into registers in the warp-major
      // geometry of radixSortOnesweepKernel: each warp handles a contiguous
      // span of the tile, split into consecutive rounds of warpSize elements
      Value values[ itemsPerThread ]{};
      unsigned char digits[ itemsPerThread ]{};
      #pragma unroll
      for( int j = 0; j < itemsPerThread; j++ ) {
         const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
         if( idxInTile < n ) {
            values[ j ] = s_tiles[ cur ][ idxInTile ];
            digits[ j ] = static_cast< unsigned char >( radixSortDigit< Value, descending >( values[ j ], pass ) );
         }
      }

      // rank the items (identical to radixSortOnesweepKernel): the lowest
      // lane of each group of equal digits atomically adds the group size to
      // the running (warp, digit) offset and broadcasts the returned value
      // within the group, which gives the offset of the item's round within
      // the warp
      int itemOffsets[ itemsPerThread ]{};
      #pragma unroll
      for( int j = 0; j < itemsPerThread; j++ ) {
         // active lanes of the warp form a prefix of the warp, so the mask
         // of lanes participating in the warp match can be computed directly
         const int activeInWarp = TNL::max(
            0,
            TNL::min(
               static_cast< int >( warpSize ),
               n - warpId * ( warpSize * itemsPerThread ) - j * static_cast< int >( warpSize ) ) );
         const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
         if( activeInWarp > 0 && idxInTile < n ) {
            const RadixSortWarpMask activeMask =
               activeInWarp >= static_cast< int >( warpSize )
                  ? fullMask
                  : ( static_cast< RadixSortWarpMask >( 1 ) << activeInWarp ) - static_cast< RadixSortWarpMask >( 1 );
            const RadixSortWarpMask match = __match_any_sync( activeMask, static_cast< int >( digits[ j ] ) );
            const int rank = radixSortPopcount( match & laneMaskLt );
            volatile int* counter = &s_warpOffsets[ warpId * radixSortBuckets + digits[ j ] ];
            const int roundOffset = *counter;
            __syncwarp( activeMask );
            if( rank == 0 )
               *counter = roundOffset + radixSortPopcount( match );
            __syncwarp( activeMask );
            itemOffsets[ j ] = roundOffset + rank;
         }
      }
      __syncthreads();

      // exclusive scan of the warp offsets of each digit (in place), with
      // one thread per digit, and the per-digit counts into s_digitStart
      int digitCount = 0;
      #pragma unroll
      for( int w = 0; w < radixSortMaxWarps; w++ ) {
         const int idx = w * radixSortBuckets + digit;
         const int count = s_warpOffsets[ idx ];
         s_warpOffsets[ idx ] = digitCount;
         digitCount += count;
      }
      s_digitStart[ digit + 1 ] = digitCount;
      __syncthreads();

      // turn the digit counts into the tile-local starts of the digit runs
      // by a raking scan of the first warp (identical to
      // radixSortOnesweepKernel), and detect the single-digit case
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

         // if all items fall into the same digit bin, the bin is recorded so
         // that the permute (an identity permutation) can be skipped;
         // at most one bin can own all items, since the counts sum to the
         // tile size, so the write below has a single writer
         int single = -1;
         #pragma unroll
         for( int k = 0; k < digitsPerThread; k++ )
            if( counts[ k ] == n )
               single = lane * digitsPerThread + k;
         const RadixSortWarpMask ballot = __ballot_sync( fullMask, single >= 0 );
         if( single >= 0 )
            s_singleDigit = single;
         else if( lane == 0 && ballot == 0 )
            s_singleDigit = -1;
      }
      __syncthreads();

      // a single-digit pass is the identity permutation: skip the permute
      // and keep the current tile (the flip is uniform across the block)
      if( s_singleDigit < 0 ) {
         // fold the digit start and the warp offset of each item into its
         // rank offset, which turns it into the position in the other tile
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ ) {
            const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
            if( idxInTile < n )
               itemOffsets[ j ] += s_digitStart[ digits[ j ] ] + s_warpOffsets[ warpId * radixSortBuckets + digits[ j ] ];
         }

         // stage the items into the other tile, grouped by digit
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ ) {
            const int idxInTile = warpId * ( warpSize * itemsPerThread ) + j * static_cast< int >( warpSize ) + lane;
            if( idxInTile < n )
               s_tiles[ cur ^ 1 ][ itemOffsets[ j ] ] = values[ j ];
         }

         // flip the tiles for the next pass (uniform across the block)
         cur ^= 1;
      }
      __syncthreads();
   }

   // write the final tile back with coalesced stores
   for( int i = digit; i < n; i += static_cast< int >( blockDim.x ) )
      view[ i ] = s_tiles[ cur ][ i ];
#endif
}

/**
 * \brief Workspace of the temporary buffers of the radix sort driver.
 *
 * The radix sort driver needs an auxiliary array of the size of the input
 * (the ping-pong buffer of the sweeps), the digit histograms of the passes,
 * the tile status counters of the decoupled lookback and the pass flags.
 * A default-constructed workspace owns no memory; \ref setSize grows the
 * buffers to the requested size and they are never shrunk, so a workspace
 * reused across sorts performs no device memory allocation at all once the
 * buffers fit the input. The auxiliary array is the dominant buffer (one
 * value per element) and its allocation costs several microseconds per
 * sort even when the memory is cached by the GPU driver.
 *
 * The value, device and index types of the buffers must match the sorted
 * array, which is enforced by the `sort` overloads of \ref RadixSort
 * taking the workspace.
 *
 * \note Like the lookback states cache of the GPU scan (see
 * \ref TNL::Algorithms::detail::lookbackStatesCache), the workspace carries
 * state between asynchronous kernel launches, so the sorts sharing one
 * workspace must be ordered on the same stream and must not overlap.
 *
 * \tparam Value is the value type of the arrays to be sorted.
 * \tparam Device is the device type, only \ref Devices::GPU is implemented.
 * \tparam Index is the index type of the arrays to be sorted.
 */
template< typename Value, typename Device, typename Index >
struct RadixSortWorkspace
{
   static_assert( std::is_same_v< Device, Devices::GPU >, "RadixSort is implemented only for the GPU device" );
   static_assert( std::is_arithmetic_v< Value >, "RadixSort supports only arithmetic value types" );
   static_assert( sizeof( Value ) <= 8, "RadixSort supports value types with at most 8 bytes" );

   /**
    * \brief Grows the buffers so that an array of `size` elements can be
    * sorted.
    *
    * The buffers keep their capacity when `size` shrinks - only the
    * buffers which cannot hold a `size`-element array are reallocated.
    */
   void
   setSize( const Index size )
   {
      // the sweeps ping-pong the whole array, so the auxiliary array must
      // hold the whole input; it is by far the largest buffer
      if( aux.getSize() < size )
         aux.setSize( size );

      // the sizes of the histogram and flag buffers depend only on the
      // value type, so they are allocated on the first call
      constexpr int passes = static_cast< int >( sizeof( Value ) );
      if( digitBase.getSize() == 0 )
         digitBase.setSize( passes * radixSortBuckets );
      if( passFlags.getSize() == 0 )
         passFlags.setSize( 2 * passes + 1 );

      // one status counter per (tile, digit) plus the work counter, with
      // one such region per radix pass; the number of tiles of a strip is
      // capped by the maximum strip size of 2^28 elements, so the buffer
      // stops growing once it can hold the counters of a full strip
      if( size > 0 ) {
         // computed without rounding up to avoid overflow of the index
         // type when the size is close to its maximum
         const Index numTiles =
            size / static_cast< Index >( radixSortTileSize ) + static_cast< Index >( size % radixSortTileSize != 0 );
         const Index maxStripTiles =
            TNL::min( numTiles, ( static_cast< Index >( 1 ) << 28 ) / static_cast< Index >( radixSortTileSize ) );
         // padded to a multiple of the bucket count so that every region
         // starts 1 KiB-aligned (keep in sync with the driver in radixSort)
         const Index requiredSize = static_cast< Index >( passes ) * ( ( maxStripTiles + 1 ) * radixSortBuckets );
         if( tileStatus.getSize() < requiredSize )
            tileStatus.setSize( requiredSize );
      }
   }

   //! \brief Auxiliary (ping-pong) array of at least the sorted size.
   Containers::Array< Value, Device, Index > aux;

   //! \brief Base offsets of the digit runs: `passes` x `radixSortBuckets` entries.
   Containers::Array< Index, Device, Index > digitBase;

   //! \brief Status counters of the lookback tiles plus the work counter,
   //! with one region per radix pass (see \ref radixSort).
   Containers::Array< unsigned int, Device, Index > tileStatus;

   //! \brief Skip flags and buffer-parity flags of the passes: `2 * passes + 1` entries.
   Containers::Array< int, Device, Index > passFlags;
};

/**
 * \brief Sorts the given view using LSD radix sort, reusing the given
 * workspace of temporary buffers.
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
 * advanced between the strips (see \ref radixSortAdvanceBaseKernel). The
 * tile status counters of the lookback are laid out in the workspace as one
 * region per pass, so that a single fill can zero the counters of all
 * passes; the aggregated fill is used for arrays of at most
 * \ref radixSortAggregatedFillLimit elements, larger sorts share the first
 * region and zero it right before each sweep.
 *
 * Arrays of at most one tile (\ref radixSortTileSize elements) take a
 * separate fast path (see \ref radixSortSingleTileKernel): the whole array
 * is sorted by one block in shared memory with a single kernel launch and
 * the workspace is not touched at all.
 *
 * A pass whose histogram consists of a single non-zero bucket maps all
 * elements to the same digit, so its stable partition is the identity
 * permutation and its sweeps can be skipped in all strips; the digit counts
 * are permutation-invariant, so the histograms of the later passes stay
 * valid regardless of which earlier passes are skipped. The skippable
 * passes are detected on the device (see \ref radixSortProcessHistogramsKernel),
 * together with the parity of the executed passes, which selects the source
 * and target buffer of each sweep, so the host never learns which passes
 * are going to run: all kernels are launched asynchronously on the same
 * stream and the host synchronizes only once, at the very end.
 *
 * The temporary buffers are taken from \p workspace (see
 * \ref RadixSortWorkspace), which the caller can reuse across sorts to
 * avoid the per-call device memory allocations.
 *
 * \param view is the view of the array to be sorted in place.
 * \param workspace is the scratch space whose buffers are grown to fit
 *                  `view` (they keep their capacity for the next call).
 * \tparam descending tells whether to sort in descending order.
 */
template< typename Value, typename Device, typename Index, bool descending >
void
radixSort( Containers::ArrayView< Value, Device, Index > view, RadixSortWorkspace< Value, Device, Index >& workspace )
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

   // an array of at most one tile is sorted by a single block in shared
   // memory, which replaces the whole machinery below with one kernel
   // launch and needs no temporary buffers, so the workspace is left
   // untouched (see radixSortSingleTileKernel)
   if( n <= radixSortTileSize ) {
      constexpr auto singleTileKernel = radixSortSingleTileKernel< Value, Index, descending >;
      Backend::LaunchConfiguration launch_config( 1, radixSortBlockSize );
      Backend::launchKernelAsync( singleTileKernel, launch_config, view );
      Backend::streamSynchronize( launch_config.stream );
      return;
   }

   // grow the reused buffers to fit the input; the buffers keep their
   // capacity when the input shrinks, so repeated sorts do not allocate
   workspace.setSize( n );

   // computed without rounding up to avoid overflow of n + radixSortTileSize
   // when the index type is signed and n is close to its maximum
   const Index numTiles = n / static_cast< Index >( radixSortTileSize ) + static_cast< Index >( n % radixSortTileSize != 0 );
   const int multiprocessors = Backend::getDeviceMultiprocessors( Backend::getDevice() );
   const Index maxBlocks = static_cast< Index >( multiprocessors ) * 32;

   // the sweeps of the executed passes alternate between view and the
   // auxiliary array of the workspace
   auto aux = workspace.aux.getView( 0, n );

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = radixSortBlockSize;
   launch_config.gridSize.x = static_cast< unsigned >( TNL::min( numTiles, maxBlocks ) );

   // the digit counts are independent of the order of the elements, which
   // changes between the passes, so the histograms of all passes are
   // computed upfront; the exclusive scan of each histogram then gives the
   // global base offsets of the digit runs. Everything up to the final
   // streamSynchronize is launched asynchronously on the same stream, so
   // the host never blocks.
   auto& digitBase = workspace.digitBase;
   Devices::GPU::LaunchConfiguration fillConfig;
   fillConfig.blockHostUntilFinished = false;
   Algorithms::fill< Devices::GPU >( digitBase.getData(), Index( 0 ), digitBase.getSize(), fillConfig );

   constexpr auto totalHistogramKernel = radixSortTotalHistogramKernel< Value, Index, descending >;
   Backend::launchKernelAsync( totalHistogramKernel, launch_config, view, digitBase.getView() );

   // scan the digit histograms of all passes and detect the skippable
   // passes together with the source/target buffer parity of each sweep in
   // a single kernel, so that the host can launch the sweeps of all passes
   // without knowing which of them are going to run; the fused kernel
   // replaces the per-pass non-blocking launches of the standalone
   // Algorithms::inplaceExclusiveScan
   constexpr auto processHistogramsKernel = radixSortProcessHistogramsKernel< Value, Index >;
   auto& passFlags = workspace.passFlags;
   Backend::launchKernelAsync(
      processHistogramsKernel, Backend::LaunchConfiguration( 1, radixSortBuckets ), digitBase.getView(), passFlags.getView() );

   // the input is processed in strips of up to 2^28 elements: the values
   // exchanged by the decoupled lookback must fit into 30 bits, and the
   // tile status array (half a byte per element, one counter per tile and
   // digit) stays reasonably small
   constexpr Index onesweepStripSize = ( static_cast< Index >( 1 ) << 28 );
   const Index numStrips = n / onesweepStripSize + static_cast< Index >( n % onesweepStripSize != 0 );
   const Index maxStripTiles = TNL::min( numTiles, onesweepStripSize / radixSortTileSize );

   // each pass owns one region of the tile status buffer, holding one
   // status counter per (tile, digit) of a strip plus the work counter;
   // the region size is padded to a multiple of the bucket count, so every
   // region starts on the same 1 KiB alignment as the single-region layout
   // and the per-tile counter bursts stay line-aligned in all passes;
   // the buffer is reused, so it may be larger than needed, but the kernels
   // of this call touch only the used prefix of the buffer
   auto& tileStatus = workspace.tileStatus;
   const Index tileStatusSize = ( maxStripTiles + 1 ) * radixSortBuckets;

   constexpr auto onesweepKernel = radixSortOnesweepKernel< Value, Index, descending >;
   constexpr auto advanceBaseKernel = radixSortAdvanceBaseKernel< Index >;

   // zero the tile status counters of all passes with a single fill when
   // the sort is small enough that the zeroed regions stay cache-hot until
   // their sweeps (see radixSortAggregatedFillLimit); each sweep
   // exclusively owns its pass's region of the buffer - the lookback of a
   // sweep reads only the counters published by the sweeps of the same
   // pass - so zeroing the regions of the later passes upfront is safe (the
   // fill is ordered before all sweeps on the stream); zeroing the status
   // also resets the work counter of each region
   const bool aggregateFill = n <= static_cast< Index >( radixSortAggregatedFillLimit );
   if( aggregateFill )
      Algorithms::fill< Devices::GPU >( tileStatus.getData(), 0u, static_cast< Index >( passes ) * tileStatusSize, fillConfig );

   // the host launches the sweeps of all passes: the sweeps of a skipped
   // pass exit immediately, so they permute the array by the identity
   // permutation, and the advance of the pass's base offsets by the still
   // zeroed tile counters is a no-op (see radixSortAdvanceBaseKernel)
   for( int pass = 0; pass < passes; pass++ ) {
      for( Index strip = 0; strip < numStrips; strip++ ) {
         const Index stripBegin = strip * onesweepStripSize;
         // computed from the strip size to avoid overflow of the index type
         // when n is close to its maximum (stripBegin + onesweepStripSize
         // can overflow a signed index type)
         const Index stripSize = TNL::min( n - stripBegin, onesweepStripSize );
         const Index stripEnd = stripBegin + stripSize;
         // computed without rounding up to avoid overflow of the index type
         // when the strip size is close to its maximum
         const Index stripNumTiles = stripSize / radixSortTileSize + static_cast< Index >( stripSize % radixSortTileSize != 0 );

         // with the aggregated fill the sweep uses its pass's region;
         // without it the sweeps share the first region (zeroed right
         // before each sweep, so that it is still cache-hot when the sweep
         // runs), which keeps the counter addressing identical to the
         // single-region layout; the strips after the first one always
         // re-zero the region (also resetting the work counter)
         const Index regionBegin = aggregateFill ? pass * tileStatusSize : 0;
         auto tileStatusRegion = tileStatus.getView( regionBegin, regionBegin + tileStatusSize );
         if( ! aggregateFill || strip > 0 )
            Algorithms::fill< Devices::GPU >( tileStatus.getData() + regionBegin, 0u, tileStatusSize, fillConfig );
         launch_config.gridSize.x = static_cast< unsigned >( TNL::min( stripNumTiles, maxBlocks ) );
         Backend::launchKernelAsync(
            onesweepKernel,
            launch_config,
            view,
            aux,
            digitBase.getView(),
            tileStatusRegion,
            passFlags.getConstView(),
            stripBegin,
            stripEnd,
            stripNumTiles,
            pass );

         // the last tile of the strip has published the strip's digit totals
         // in its tile counters, which advance the base offsets for the next
         // strip
         if( strip + 1 < numStrips ) {
            launch_config.gridSize.x = 1;
            Backend::launchKernelAsync(
               advanceBaseKernel, launch_config, digitBase.getView(), tileStatusRegion, stripNumTiles, pass );
         }
      }
   }

   // for an odd number of executed passes the sorted data ends up in the
   // auxiliary array and is copied back by the kernel; zero executed passes
   // means that all elements share every digit, i.e. the input is already
   // sorted in place (even parity, no copy)
   constexpr auto copyBackKernel = radixSortCopyBackKernel< Value, Index >;
   launch_config.gridSize.x = static_cast< unsigned >( TNL::min( numTiles, maxBlocks ) );
   Backend::launchKernelAsync( copyBackKernel, launch_config, view, aux, passFlags.getConstView() );
   Backend::streamSynchronize( launch_config.stream );
#else
   throw Exceptions::NotImplementedError( "RadixSort is supported only when CUDA or HIP is enabled." );
#endif
}

/**
 * \brief One-shot overload of \ref radixSort which allocates a fresh
 * workspace per call.
 *
 * Repeated sorts should prefer the overload taking \ref RadixSortWorkspace,
 * which removes the per-call device memory allocations.
 */
template< typename Value, typename Device, typename Index, bool descending >
void
radixSort( Containers::ArrayView< Value, Device, Index > view )
{
   RadixSortWorkspace< Value, Device, Index > workspace;
   radixSort< Value, Device, Index, descending >( view, workspace );
}

}  // namespace TNL::Algorithms::Sorting::detail
