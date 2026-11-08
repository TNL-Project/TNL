// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <type_traits>
#include <utility>

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

//! \brief Number of threads per block in the histogram, per-digit and
//! single-tile kernels of the radix sort.
//!
//! The per-digit kernels (\ref radixSortProcessHistogramsKernel and
//! \ref radixSortAdvanceBaseKernel) rely on each thread handling exactly one
//! digit, which requires the block size to be equal to the number of digit
//! buckets. The histogram and single-tile kernels use the same block size
//! for simplicity.
constexpr int radixSortBlockSize = 256;

//! \brief Number of elements processed by one thread in one tile of the
//! histogram and single-tile kernels.
//!
//! Processing multiple elements per thread amortizes the shared-memory
//! zeroing, the per-digit scan and the synchronization over more elements
//! and provides more independent loads to hide the memory latency.
constexpr int radixSortItemsPerThread = 8;

//! \brief Number of elements processed by one tile of the single-tile
//! kernel.
constexpr int radixSortTileSize = radixSortBlockSize * radixSortItemsPerThread;

//! \brief Number of threads per block of \ref radixSortOnesweepKernel.
//!
//! Unlike the per-digit kernels, the sweep kernel does not require the block
//! size to be equal to the number of digit buckets: its per-digit phases
//! (lookback, digit starts) are guarded to the first \ref radixSortBuckets
//! threads and the per-\e word phases to the first `radixSortBuckets / 2`
//! threads. Larger blocks spread the barrier, lookback and zeroing costs
//! over more elements and improve the number of resident warps.
constexpr int radixSortOnesweepBlockSize = 384;

//! \brief Number of elements processed by one thread in one tile of
//! \ref radixSortOnesweepKernel.
//!
//! The items fill the staging tile to about 24 KiB of shared memory for any
//! value type: the remaining shared-memory budget of a thread block (the
//! per-warp rank counters and the digit bases) then keeps three blocks
//! resident on a streaming multiprocessor for both 4-byte and 8-byte values.
template< typename Value >
constexpr int radixSortOnesweepItemsPerThread = ( sizeof( Value ) <= 4 ) ? 16 : 8;

//! \brief Number of elements processed by one tile of
//! \ref radixSortOnesweepKernel (6,144 for values of at most 4 bytes,
//! 3,072 for 8-byte values).
template< typename Value >
constexpr int radixSortOnesweepTileSize = radixSortOnesweepBlockSize * radixSortOnesweepItemsPerThread< Value >;

//! \brief Number of threads per block in the medium-tile configuration of
//! \ref radixSortOnesweepKernel.
constexpr int radixSortOnesweepMediumBlockSize = 512;

//! \brief Number of elements processed by one thread in the medium-tile
//! configuration of \ref radixSortOnesweepKernel.
constexpr int radixSortOnesweepMediumItemsPerThread = 8;

//! \brief Number of elements in one medium tile of \ref radixSortOnesweepKernel.
constexpr int radixSortOnesweepMediumTileSize = radixSortOnesweepMediumBlockSize * radixSortOnesweepMediumItemsPerThread;

/**
 * \brief Array size threshold below which the sweep uses the small tile
 * configuration of the histogram and single-tile kernels
 * (\ref radixSortTileSize).
 *
 * The 2,048-element tiles maximize block parallelism at small sizes. Above
 * this threshold, the 4,096-element medium tiles reduce the per-tile ranking
 * and look-back costs without an occupancy regression. The measured crossover
 * is at about 100,000 elements for values of at most four bytes, but between
 * 150,000 and 175,000 elements for eight-byte values.
 */
template< typename Value >
constexpr int radixSortOnesweepSmallTileLimit = ( sizeof( Value ) <= 4 ) ? 100000 : 160000;

/**
 * \brief Array size threshold below which the sweep uses medium tiles.
 *
 * Above this threshold, the deeper ranking phases of the big configuration
 * are amortized over enough elements to outweigh its lower block parallelism.
 */
constexpr int radixSortOnesweepMediumTileLimit = 1 << 18;

//! \brief Number of elements processed by one tile of
//! \ref radixSortOnesweepKernel for the given array size.
template< typename Value, typename Index >
constexpr Index
radixSortOnesweepTileSizeFor( const Index n )
{
   if( n <= static_cast< Index >( radixSortOnesweepSmallTileLimit< Value > ) )
      return static_cast< Index >( radixSortTileSize );
   if( n <= static_cast< Index >( radixSortOnesweepMediumTileLimit ) )
      return static_cast< Index >( radixSortOnesweepMediumTileSize );
   return static_cast< Index >( radixSortOnesweepTileSize< Value > );
}

//! \brief Array size threshold below which the tile status regions of all
//! passes are zeroed by a single upfront fill.
//!
//! The aggregated fill replaces one fill launch per pass, but it pays off
//! only while the sweeps' streaming traffic between the fill and the last
//! pass stays small enough for the zeroed regions to stay cache-hot (at
//! about 2^22 elements on the current GPUs); larger sorts share the first
//! region and zero it right before each sweep (see \ref radixSort).
constexpr int radixSortAggregatedFillLimit = 1 << 22;

//! \brief Array size threshold below which the sorts replay their captured
//! kernel launch sequence through a cached CUDA/HIP graph.
//!
//! Replaying the launch sequence removes the host-side launch overhead,
//! which is exposed only while the kernels are so short that the device
//! drains the launch queue faster than the host fills it. Once the kernels
//! are long enough, the host issues the whole sequence before the device
//! runs out of queued work, so the direct launches hide their own overhead
//! while the replayed graph still pays the device-side node scheduling
//! between the kernels (measured at about 2% at one million elements). The
//! limit is set below the regression - the largest measured winning size
//! was 100k elements.
constexpr int radixSortGraphLimit = 1 << 19;

//! \brief Maximum array size for programmatic dependent launches.
//!
//! Small sorts can hide launch latency and drain their short grids together;
//! larger sorts already hide host launch submission behind device execution.
//! The limit matches CUB's radix-sort policy and avoids deeply nesting
//! dependent grids for very large inputs.
constexpr int radixSortPdlLimit = 1 << 24;

// the per-digit kernels are launched with one thread per bucket (see the
// comment on radixSortBlockSize)
static_assert( radixSortBlockSize == radixSortBuckets, "radixSortBlockSize must be equal to radixSortBuckets" );

//! \brief Maximum number of warps per block in the single-tile kernel,
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
//! \note The warp match intrinsic used by the single-tile kernel and by
//! the HIP fallback of \ref radixSortWarpMatchAny requires compute
//! capability 7.0 or newer on CUDA and gfx8 or newer on HIP.
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
 * \brief Returns the mask of the lanes holding the same label as the calling
 * lane, i.e. the equivalent of `__match_any_sync` invoked with the full warp
 * mask.
 *
 * On CUDA, the match is computed by a sequence of `radixSortBits` warp
 * ballots, one per bit of the label, instead of the native match intrinsic:
 * the native `match.any.sync` instruction has a long dependent latency which
 * stalls the instruction issue of the re-ranking loop of
 * \ref radixSortOnesweepKernel (it was measured as the dominant bottleneck
 * of the sweep kernel), while the ballot sequence interleaves with the
 * surrounding instructions much better. This is the same trade-off made by
 * CUB's warp matcher (`cub/util_ptx.cuh`). On HIP, the native intrinsic is
 * kept: the ballot sequence is written in CUDA's inline PTX and the measured
 * latency pathology is specific to the CUDA GPUs of the current generation.
 *
 * All lanes of the warp must participate in the call - the binning kernel
 * pads the partial tiles so that its warp collectives always run on full
 * warps (see \ref radixSortOnesweepKernel).
 *
 * \param label is the label whose equal-lane peers are searched for; only
 *              the low `radixSortBits` bits are significant.
 */
[[nodiscard]] __device__
__forceinline__ RadixSortWarpMask
radixSortWarpMatchAny( unsigned int label )
{
   #if defined( __CUDACC__ )
   RadixSortWarpMask match = 0;
   #pragma unroll
   for( int bit = 0; bit < radixSortBits; bit++ ) {
      unsigned int mask;
      const unsigned int currentBit = 1u << bit;
      asm(
         "{\n"
         "   .reg .pred p;\n"
         "   and.b32 %0, %1, %2;\n"
         "   setp.ne.u32 p, %0, 0;\n"
         "   vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
         "   @!p not.b32 %0, %0;\n"
         "}\n"
         : "=r"( mask )
         : "r"( label ), "r"( currentBit ) );
      if( bit == 0 )
         match = mask;
      else
         match &= mask;
   }
   return match;
   #else
   return __match_any_sync( Backend::getWarpFullMask(), static_cast< int >( label ) );
   #endif
}

/**
 * \brief Returns the index of the lowest set bit of a warp synchronization
 * mask (-1 when the mask is empty).
 */
[[nodiscard]] __device__
__forceinline__ int
radixSortWarpFirstLane( RadixSortWarpMask mask )
{
   if constexpr( sizeof( RadixSortWarpMask ) == 4 )
      return __ffs( static_cast< int >( mask ) ) - 1;
   else
      return __ffsll( static_cast< long long >( mask ) ) - 1;
}

/**
 * \brief Atomically adds `count` to the 16-bit counter of the given digit
 * packed with another digit's counter in a 32-bit shared word, and returns
 * the previous value of the digit's counter.
 *
 * The counters of each (warp, digit) pair are packed two per 32-bit
 * word, which halves the shared-memory footprint of the rank counters of
 * \ref radixSortOnesweepKernel and is what allows three thread blocks to
 * stay resident on a streaming multiprocessor. Neither half of a word can
 * overflow 16 bits (a digit's count and running offset stay below the tile
 * size of the sweep), so a shifted 32-bit atomic addition never carries
 * into the other half. On devices without a 16-bit shared-memory atomic add
 * (that is, on all of them) this is also the only portable formulation.
 *
 * \param packedCounters points to the shared word holding the digit's
 *                       counter in its `16 * (digit & 1)`-shifted half.
 * \param digit is the digit whose counter is incremented.
 * \param count is the increment (at most the warp size).
 */
[[nodiscard]] __device__
__forceinline__ unsigned int
radixSortPackedCounterAdd( unsigned int* packedCounters, unsigned int digit, unsigned int count )
{
   const unsigned int shift = 16u * ( digit & 1u );
   return ( atomicAdd( packedCounters, count << shift ) >> shift ) & 0xFFFFu;
}

#endif

/**
 * \brief Returns the value used to fill the rounded-up rounds of partial
 * tiles in \ref radixSortOnesweepKernel.
 *
 * The staged positions of the padded lanes are forced to the last bucket
 * during the ranking, so the padding value is never written to the sorted
 * array; any value with an in-range digit would work, and a NaN is used for
 * floating-point types because its raw bytes mark it clearly as never
 * touched when it does leak into observability during debugging.
 */
template< typename Value, bool descending >
__cuda_callable__
Value
radixSortPaddingValue()
{
   if constexpr( std::is_floating_point_v< Value > ) {
      // ascending: key of all ones comes from bits 0x7FF...F (a positive
      // NaN); descending: a key of zero comes from bits of all ones (a
      // negative NaN, inverted to zero by the key mapping)
#if defined( __CUDA_ARCH__ ) || defined( __HIP__ )
      if constexpr( sizeof( Value ) == 4 )
         return __uint_as_float( descending ? 0xFFFFFFFFu : 0x7FFFFFFFu );
      else
         return __longlong_as_double(
            descending ? static_cast< long long >( 0xFFFFFFFFFFFFFFFFull )
                       : static_cast< long long >( 0x7FFFFFFFFFFFFFFFull ) );
#else
      using Key = RadixSortKey< Value >;
      constexpr Key signBit = Key( 1 ) << ( 8 * sizeof( Value ) - 1 );
      const Key bits = descending ? ~Key( 0 ) : ( ~Key( 0 ) & ~signBit );
      unsigned char raw[ sizeof( Value ) ];
      for( int byte = 0; byte < static_cast< int >( sizeof( Value ) ); byte++ )
         raw[ byte ] = static_cast< unsigned char >( ( bits >> ( 8 * byte ) ) & 0xFF );
      Value value;
      std::memcpy( &value, raw, sizeof( Value ) );
      return value;
#endif
   }
   else {
      using Key = RadixSortKey< Value >;
      constexpr Key signBit = Key( 1 ) << ( 8 * sizeof( Value ) - 1 );
      const Key bits = ( std::is_signed_v< Value > ? signBit : Key( 0 ) ) ^ ( descending ? Key( 0 ) : ~Key( 0 ) );
      Value value;
      auto* out = reinterpret_cast< unsigned char* >( &value );
      for( int byte = 0; byte < static_cast< int >( sizeof( Value ) ); byte++ )
         out[ byte ] = static_cast< unsigned char >( ( bits >> ( 8 * byte ) ) & 0xFF );
      return value;
   }
}

#if defined( __CUDACC__ ) || defined( __HIP__ )

/**
 * \brief Shared-memory union of the tile-local digit starts and the staging
 * tile of \ref radixSortOnesweepKernel.
 *
 * The digit starts are folded into the running per-(warp, digit) rank
 * counters and into the shifted digit bases before the re-ranking (and the
 * lookback) consumes them for the last time, so the staging tile written
 * afterwards can share their storage, keeping the block's shared memory
 * low enough for three resident blocks per streaming multiprocessor.
 */
template< typename Value, int blockThreads, int itemsPerThread >
union RadixSortOnesweepStage
{
   //! \brief Tile-local starts of the digit runs (`digitStarts[d]` is the
   //! number of tile items with a digit smaller than `d`).
   unsigned short digitStarts[ radixSortBuckets + 1 ];

   //! \brief Staging tile of the scattered items, grouped by digit.
   Value tile[ blockThreads * itemsPerThread ];
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

//! \brief Pass-flag value marking a skip flag not yet published by the
//! current launch of \ref radixSortProcessHistogramsKernel.
constexpr int radixSortPassPending = -1;

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
 * \param flags is the view of the pass flags; its skip slots are initialized
 *              to \ref radixSortPassPending for the processing kernel.
 */
template< typename Value, typename Index, bool descending >
__global__
void
radixSortTotalHistogramKernel(
   const Containers::ArrayView< Value, Devices::GPU, Index > input,
   Containers::ArrayView< Index, Devices::GPU, Index > counts,
   Containers::ArrayView< int, Devices::GPU, Index > flags )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Backend::triggerProgrammaticLaunchCompletion();
   constexpr int passes = static_cast< int >( sizeof( Value ) );
   // the pending values distinguish this launch's skip flags from flags left
   // by a previous sort; the processing kernel starts after this whole grid
   if( blockIdx.x == 0 && static_cast< int >( threadIdx.x ) < passes )
      flags[ threadIdx.x ] = radixSortPassPending;

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

   // The input and shared histograms are independent of the zero fill, but
   // the global accumulation consumes its zeroed counters.
   Backend::gridDependencySynchronize();
   for( int i = static_cast< int >( threadIdx.x ); i < passes * radixSortBuckets; i += static_cast< int >( blockDim.x ) )
      radixSortAtomicAdd( &counts[ i ], static_cast< Index >( s_counts[ i / radixSortBuckets ][ i % radixSortBuckets ] ) );
#endif
}

/**
 * \brief Kernel scanning the digit histograms and deriving the skippable
 * passes and the sweep buffers from them.
 *
 * The global base offsets of the digit runs are the exclusive scans of the
 * per-pass histograms. The launch uses one block of \ref radixSortBuckets
 * threads per pass, so the block-wide exclusive scans of all (at most 8)
 * passes run concurrently and are written back in place.
 *
 * A pass whose histogram consists of a single non-zero bucket maps all
 * elements to the same digit, so the stable partition of its sweep is the
 * identity permutation and the sweep can be skipped. The flags are computed
 * on the device so that the driver does not have to synchronize the host to
 * learn them: the host launches the sweeps of all passes and each sweep
 * adjusts itself (see \ref radixSortOnesweepKernel). Each block counts the
 * non-zero buckets of its pass with a block-wide reduction and publishes the
 * skip flag before starting its scan. The last block then computes the
 * running parity of the executed passes before each pass and the parity of
 * the total number of executed passes, which is a short sequential prefix.
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
 *              the parity of the total number of executed passes. The skip
 *              slots must initially hold \ref radixSortPassPending.
 */
template< typename Value, typename Index >
__global__
void
radixSortProcessHistogramsKernel(
   Containers::ArrayView< Index, Devices::GPU, Index > counts,
   Containers::ArrayView< int, Devices::GPU, Index > flags )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Backend::gridDependencySynchronize();
   Backend::triggerProgrammaticLaunchCompletion();
   constexpr int passes = static_cast< int >( sizeof( Value ) );
   // the launch uses blockDim.x == radixSortBuckets and one block per pass,
   // so each block holds one counter of its pass per thread
   using BlockScan =
      Algorithms::detail::GPUBlockScan< Algorithms::detail::ScanType::Exclusive, radixSortBuckets, std::plus<>, Index >;
   __shared__ typename BlockScan::Storage s_scanStorage;

   const int pass = static_cast< int >( blockIdx.x );
   const int digit = static_cast< int >( threadIdx.x );
   const Index count = counts[ static_cast< Index >( pass ) * radixSortBuckets + digit ];

   // count the non-zero buckets of the pass: a single non-zero bucket means
   // the pass maps all elements to the same digit and its sweep can be skipped
   const int nonZeroBuckets = __syncthreads_count( count != 0 );

   // publish the predicate on the raw counters before replacing them by the
   // scan, so the parity consumer never has to read another pass's counters
   if( threadIdx.x == 0 )
      atomicExch( &flags[ pass ], static_cast< int >( nonZeroBuckets == 1 ) );

   const Index base = BlockScan::scan( std::plus<>{}, Index( 0 ), count, static_cast< int >( threadIdx.x ), s_scanStorage );
   counts[ static_cast< Index >( pass ) * radixSortBuckets + digit ] = base;

   // gather the predicates concurrently after the scan, when the other blocks
   // have already had the scan interval to publish them; there are at most 8
   // flags, so one warp ballot collects them without another block barrier
   if( pass == passes - 1 && threadIdx.x < warpSize ) {
      int skip = 0;
      if( static_cast< int >( threadIdx.x ) < passes ) {
         const volatile int* const skipFlag = &flags[ threadIdx.x ];
         do {
            skip = *skipFlag;
         } while( skip == radixSortPassPending );
      }
      const RadixSortWarpMask skipMask = __ballot_sync( Backend::getWarpFullMask(), skip );

      // the parity prefix is sequential over the (at most 8) passes, so it is
      // computed by a single lane
      if( threadIdx.x == 0 ) {
         int parity = 0;
         #pragma unroll
         for( int p = 0; p < passes; p++ ) {
            flags[ passes + p ] = parity;
            parity ^= 1 - static_cast< int >( ( skipMask >> p ) & 1 );
         }
         flags[ 2 * passes ] = parity;
      }
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

// the launch bounds keep the occupancy of the binning kernel from silently
// dropping when a change grows the register count: the medium configuration
// (512 threads, 2 blocks per multiprocessor = 1,024 of 1,536 threads) allows
// at most 64 registers per thread, and the big configuration (384 threads,
// 3 blocks = 1,152 threads) allows at most 56; their shared-memory usage
// stays below the respective block budgets as well; HIP is not guarded
// because the attribute was not verified there
#if defined( __CUDACC__ )
   #define RADIX_SORT_LAUNCH_BOUNDS( blockThreads, itemCount ) \
      __launch_bounds__(                                       \
         blockThreads,                                         \
         blockThreads == radixSortOnesweepMediumBlockSize ? 2  \
         : blockThreads == radixSortOnesweepBlockSize     ? 3  \
                                                          : ( sizeof( Value ) <= 4 ? 4 : 2 ) )
#else
   #define RADIX_SORT_LAUNCH_BOUNDS( blockThreads, itemCount )
#endif

/**
 * \brief Kernel performing one digit-binning pass of the single-pass variant.
 *
 * The tiles are assigned to the blocks via a work counter: an atomic
 * increment assigns the tile indices in increasing order, regardless of the
 * block scheduling. Each warp handles a contiguous span of the tile, split
 * into `radixSortOnesweepItemsPerThread< Value >` rounds of `warpSize`
 * elements, so that the loads of each round are coalesced and the
 * warp-major order of the rank offsets matches the order of the elements
 * in the tile. The tail of a partial tile is filled with
 * \ref radixSortPaddingValue: the ranking forces the padded rounds to the
 * last bucket, so all warp collectives run unmasked on full warps and no
 * per-round active-mask computation or per-item guards are needed; the
 * padded items land at the tail of the last digit's run of the staged
 * tile, past all real items, and are never written out.
 *
 * The intra-tile ranks are computed in two phases, following CUB's
 * `BlockRadixRankMatchEarlyCounts`:
 *
 * 1. Each item is atomically accumulated into its warp's shared per-digit
 *    histogram (one `atomicAdd` per item into the packed per-(warp, digit)
 *    counters, see \ref radixSortPackedCounterAdd), giving the tile's digit
 *    counts early for the lookback publishing below. The counters are then
 *    scanned in place across the warps of each digit and the block-wide
 *    digit counts are turned into the tile-local starts of the digit runs
 *    by a raking scan.
 * 2. Each item is re-ranked with a warp match, computed by
 *    \ref radixSortWarpMatchAny (O(1) in the tile size instead of a linear
 *    scan over the digits): the lowest lane
 *    of each group of equal digits atomically adds the group size to the
 *    running per-(warp, digit) counter and the returned offset is
 *    broadcast within the group, giving each item its staged position. The
 *    rank is consumed immediately by the staging write, so neither ranks
 *    nor digits are kept in registers per item (the digits are recomputed
 *    from the values whenever they are needed).
 *
 * After the ranks are consumed, the staged tile holds the items grouped by
 * digit and is written out with coalesced global stores. The per-digit
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
 * \tparam blockThreads is the block size of the tile configuration (the
 *                      big configuration of \ref radixSortOnesweepBlockSize
 *                      threads by default).
 * \tparam itemsPerThread is the number of items per thread of the tile
 *                        configuration.
 */
template<
   typename Value,
   typename Index,
   bool descending,
   int blockThreads = radixSortOnesweepBlockSize,
   int itemsPerThread = radixSortOnesweepItemsPerThread< Value > >
__global__
void
RADIX_SORT_LAUNCH_BOUNDS( blockThreads, itemsPerThread ) radixSortOnesweepKernel(
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
   Backend::gridDependencySynchronize();
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

   constexpr int tileElements = blockThreads * itemsPerThread;
   // the per-(warp, digit) counters are packed two per 32-bit word, so one
   // word covers the two digits `2 * word` and `2 * word + 1`
   constexpr int counterWords = radixSortBuckets / 2;
   constexpr int counterBlock = ( blockThreads / 32 ) * counterWords;

   __shared__ Index s_base[ radixSortBuckets ];
   __shared__ unsigned int s_warpCounters[ counterBlock ];
   __shared__ RadixSortOnesweepStage< Value, blockThreads, itemsPerThread > s_stage;
   __shared__ int s_singleDigit;
   __shared__ Index s_claimedTile;

   const int digit = static_cast< int >( threadIdx.x );

   const Index n = input.getSize();
   unsigned int* const workCounter = &tileStatus[ numTiles * radixSortBuckets ];

   const int warpId = static_cast< int >( threadIdx.x ) / warpSize;
   const int lane = static_cast< int >( threadIdx.x ) % warpSize;
   const int warps = static_cast< int >( blockDim.x ) / warpSize;
   const RadixSortWarpMask fullMask = Backend::getWarpFullMask();
   const RadixSortWarpMask laneMaskLt =
      ( static_cast< RadixSortWarpMask >( 1 ) << lane ) - static_cast< RadixSortWarpMask >( 1 );

   // the rounded-up rounds of a partial tile are filled with the padding
   // value (its digit is the last bucket in every pass), so the warp
   // collectives below always run unmasked on full warps
   const Value padValue = radixSortPaddingValue< Value, descending >();

   // the work counter assigns tile indices in increasing order, so the block
   // processing a tile can rely on all its predecessors being claimed
   while( true ) {
      if( threadIdx.x == 0 )
         s_claimedTile = static_cast< Index >( atomicAdd( workCounter, 1u ) );
      __syncthreads();
      const Index tile = s_claimedTile;
      if( tile >= numTiles )
         break;
      const Index tileBegin = tile * static_cast< Index >( tileElements );
      const Index tileEnd = tileBegin + TNL::min( n - tileBegin, static_cast< Index >( tileElements ) );
      const int tileSize = static_cast< int >( tileEnd - tileBegin );
      const int padCount = tileElements - tileSize;

      // the loads and stores of the tile are addressed with 32-bit
      // tile-relative indices from the base pointer of the tile
      const Value* const tileInput = input.getData() + tileBegin;

      // load the items into registers in the warp-major layout (each warp
      // handles a contiguous span of the tile, split into consecutive
      // rounds of warpSize elements, so that the loads of each round are
      // coalesced); all loads of the tile are issued before any shared
      // memory access: the compiler does not move a global load across a
      // shared atomic, so loads interleaved with the histogram atomics
      // below would wait for the full memory latency one after another,
      // and issuing them before the zeroing barrier also overlaps their
      // latency with the zeroing; the rounded-up rounds of a partial tile
      // are filled with the padding value
      Value values[ itemsPerThread ];
      const int warpBase = warpId * ( warpSize * itemsPerThread );
      if( tileSize == tileElements ) {
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ )
            values[ j ] = tileInput[ warpBase + j * static_cast< int >( warpSize ) + lane ];
      }
      else {
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ ) {
            const int idxInTile = warpBase + j * static_cast< int >( warpSize ) + lane;
            values[ j ] = idxInTile < tileSize ? tileInput[ idxInTile ] : padValue;
         }
      }

      // zero the packed per-(warp, digit) counters of this tile and the
      // anchor of the digit starts (the staging tile of the previous tile
      // of this block aliases the digit starts and may have clobbered it)
      for( int i = digit; i < warps * counterWords; i += static_cast< int >( blockDim.x ) )
         s_warpCounters[ i ] = 0;
      s_stage.digitStarts[ 0 ] = 0;
      __syncthreads();

      // accumulate each item into its warp's shared per-digit histogram;
      // the first phase of the ranking ends here, giving the tile's digit
      // counts early for the lookback publishing below
      if( tileSize == tileElements ) {
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ ) {
            const unsigned int dig = radixSortDigit< Value, descending >( values[ j ], pass );
            (void) radixSortPackedCounterAdd( &s_warpCounters[ warpId * counterWords + ( dig >> 1 ) ], dig, 1 );
         }
      }
      else {
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ ) {
            const int idxInTile = warpBase + j * static_cast< int >( warpSize ) + lane;
            // the rounded-up rounds go to the last bucket regardless of the
            // padding value (which lands in the staged tile but is never
            // written out)
            const unsigned int dig =
               idxInTile < tileSize ? radixSortDigit< Value, descending >( values[ j ], pass ) : radixSortBuckets - 1;
            (void) radixSortPackedCounterAdd( &s_warpCounters[ warpId * counterWords + ( dig >> 1 ) ], dig, 1 );
         }
      }
      __syncthreads();

      // exclusive scan of the packed per-warp counters of each digit pair
      // (in place, one thread per counter word); the packed counts of the
      // digits stay in registers for the lookback publishing and the digit
      // starts written below
      if( digit < counterWords ) {
         unsigned int exclusive = 0;
         for( int w = 0; w < warps; w++ ) {
            const int idx = w * counterWords + digit;
            const unsigned int count = s_warpCounters[ idx ];
            s_warpCounters[ idx ] = exclusive;
            exclusive += count;
         }
         // the last digit's count includes the padding items; it is fixed
         // before publishing so that the exchanged strip counters hold only
         // the counts of the real items (the staging below still uses the
         // padded counts, which place the padding at the tail of the last
         // digit's run)
         const int countLow = static_cast< int >( exclusive & 0xFFFFu );
         const int countHigh = static_cast< int >( exclusive >> 16 );
         s_stage.digitStarts[ 2 * digit + 1 ] = static_cast< unsigned short >( countLow );
         s_stage.digitStarts[ 2 * digit + 2 ] = static_cast< unsigned short >( countHigh );
         const int realLow = countLow;
         const int realHigh = countHigh - ( 2 * digit + 1 == radixSortBuckets - 1 ? padCount : 0 );
         // publish the tile's aggregate counts before waiting on any
         // predecessor, which guarantees that the lookback of all blocks
         // always makes progress
         atomicExch(
            &tileStatus[ tile * radixSortBuckets + 2 * digit ],
            radixSortTileAggregate | static_cast< unsigned int >( realLow ) );
         atomicExch(
            &tileStatus[ tile * radixSortBuckets + 2 * digit + 1 ],
            radixSortTileAggregate | static_cast< unsigned int >( realHigh ) );
      }
      __syncthreads();

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
            counts[ k ] = s_stage.digitStarts[ lane * digitsPerThread + k + 1 ];
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
            s_stage.digitStarts[ lane * digitsPerThread + k + 1 ] = static_cast< unsigned short >( running );
         }

         // if all items of the tile fall into the same digit bin, the bin
         // is recorded so that the tile can be written out without
         // staging; at most one bin can own all items (the counts include
         // the padding, so only the last bin can reach the padded tile
         // size of a partial tile), so the write below has a single writer
         int single = -1;
         #pragma unroll
         for( int k = 0; k < digitsPerThread; k++ )
            if( counts[ k ] == tileElements )
               single = lane * digitsPerThread + k;
         const RadixSortWarpMask ballot = __ballot_sync( fullMask, single >= 0 );
         if( single >= 0 )
            s_singleDigit = single;
         else if( lane == 0 && ballot == 0 )
            s_singleDigit = -1;
      }
      __syncthreads();

      // the lookback below needs the tile-local start and end of the
      // thread's digit run, which are kept in registers: the lookback is
      // deferred past the staging, whose tile clobbers the digit starts
      // (see the union)
      int runStart = 0;
      int runEnd = 0;
      if( digit < radixSortBuckets ) {
         runStart = s_stage.digitStarts[ digit ];
         runEnd = s_stage.digitStarts[ digit + 1 ];
      }

      // fold the tile-local digit starts into the running per-(warp, digit)
      // counters (packed two per word, one thread per word); the digit
      // starts are consumed here and by the reads above, so the staging
      // tile written afterwards may clobber them (see the union)
      if( s_singleDigit < 0 && digit < counterWords ) {
         const unsigned int packedStart = static_cast< unsigned int >( s_stage.digitStarts[ 2 * digit ] )
                                        | static_cast< unsigned int >( s_stage.digitStarts[ 2 * digit + 1 ] ) << 16;
         for( int w = 0; w < warps; w++ )
            s_warpCounters[ w * counterWords + digit ] += packedStart;
      }
      __syncthreads();

      // decoupled lookback: accumulate the aggregates of the preceding
      // tiles until an inclusive prefix is found; the first tile has no
      // predecessors, so its strip-relative offset is zero; the lookback
      // runs after the tile is staged (as in CUB's onesweep), so that the
      // ranking hides the time the predecessors need to publish their
      // prefixes instead of the block stalling on them
      const auto lookback = [ & ]()
      {
         if( digit >= radixSortBuckets )
            return;
         Index stripExclusive = 0;
         if( tile != 0 ) {
            unsigned int accumulated = 0;
            bool foundPrefix = false;
            for( Index predecessor = tile; predecessor-- > 0; ) {
               unsigned int word;
               // spin until the predecessor publishes its aggregate or
               // prefix; the status and the value share one word, so no
               // acquire-release ordering is needed - the volatile load
               // bypasses L1 and the atomic exchange is performed in L2,
               // which makes the word visible in bounded time
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
         // shift the base of the digit's run by the tile-local start of
         // the run, so that the staged writeout stores with a single
         // shared operand (the start of the single-digit run is zero, so
         // the direct writeout path below is unaffected by the shift)
         s_base[ digit ] = digitBase[ static_cast< Index >( pass ) * radixSortBuckets + digit ] + stripExclusive
                         - static_cast< Index >( runStart );
         // republish with the strip-relative inclusive prefix through this
         // tile, which shortens the lookback of the subsequent blocks; the
         // real count is recovered from the raked starts (the last digit's
         // count includes the padding, which does not belong to the strip)
         const int realCount = runEnd - runStart - ( digit == radixSortBuckets - 1 ? padCount : 0 );
         // the block of the last tile of the strip publishes the strip's
         // total count of its digit in this counter, from which the driver
         // advances the base offsets of the digit runs for the next strip
         atomicExch(
            &tileStatus[ tile * radixSortBuckets + digit ],
            radixSortTilePrefix | static_cast< unsigned int >( stripExclusive + realCount ) );
      };

      if( s_singleDigit >= 0 ) {
         lookback();
         __syncthreads();
         // all items of the tile fall into the same digit bin: write them
         // out directly from registers, skipping the staged tile; the
         // intra-tile rank of an item is its index in the tile, so the
         // stores are stable and consecutive
         #pragma unroll
         for( int j = 0; j < itemsPerThread; j++ ) {
            const int idxInTile = warpBase + j * static_cast< int >( warpSize ) + lane;
            if( idxInTile < tileSize )
               output[ s_base[ s_singleDigit ] + idxInTile ] = values[ j ];
         }
      }
      else {
         // re-rank the items and stage them into the shared tile in a
         // single pass: the lowest lane of each group of equal digits (the
         // lane with no lower-lane peer) atomically adds the group size to
         // the running per-(warp, digit) counter and the returned offset,
         // broadcast within the group, gives each item its staged position
         // directly - the rank is consumed by the staging write, so no
         // per-item rank or digit arrays are kept in registers (the
         // warp-crossing between the rounds is ordered by the shared
         // counter: rounds of a warp are issued in order and the running
         // counters of each warp are private to it); the rounded-up rounds
         // of a partial tile are forced to the last bucket regardless of
         // the padding value
         if( tileSize == tileElements ) {
            #pragma unroll
            for( int j = 0; j < itemsPerThread; j++ ) {
               const unsigned int dig = radixSortDigit< Value, descending >( values[ j ], pass );
               const RadixSortWarpMask match = radixSortWarpMatchAny( dig );
               const int rank = radixSortPopcount( match & laneMaskLt );
               int staged = 0;
               if( rank == 0 )
                  staged = static_cast< int >( radixSortPackedCounterAdd(
                     &s_warpCounters[ warpId * counterWords + ( dig >> 1 ) ], dig, radixSortPopcount( match ) ) );
               staged = __shfl_sync( fullMask, staged, radixSortWarpFirstLane( match ) );
               s_stage.tile[ staged + rank ] = values[ j ];
            }
         }
         else {
            #pragma unroll
            for( int j = 0; j < itemsPerThread; j++ ) {
               const int idxInTile = warpBase + j * static_cast< int >( warpSize ) + lane;
               const unsigned int dig = idxInTile < tileSize ? radixSortDigit< Value, descending >( values[ j ], pass )
                                                             : static_cast< unsigned int >( radixSortBuckets - 1 );
               const RadixSortWarpMask match = radixSortWarpMatchAny( dig );
               const int rank = radixSortPopcount( match & laneMaskLt );
               int staged = 0;
               if( rank == 0 )
                  staged = static_cast< int >( radixSortPackedCounterAdd(
                     &s_warpCounters[ warpId * counterWords + ( dig >> 1 ) ], dig, radixSortPopcount( match ) ) );
               staged = __shfl_sync( fullMask, staged, radixSortWarpFirstLane( match ) );
               s_stage.tile[ staged + rank ] = values[ j ];
            }
         }
         lookback();
         __syncthreads();

         // write the tile out with coalesced stores: consecutive positions
         // of the staged tile belong to the same digit run (or to adjacent
         // runs), so the global stores are consecutive as well; the digit
         // owning each position is recomputed from the staged value
         // instead of searching the digit starts - it is a pure function
         // of the value and the pass; the padding stays past the written
         // positions of the last digit's run and is never written out
         for( int i = digit; i < tileSize; i += static_cast< int >( blockDim.x ) ) {
            const Value value = s_stage.tile[ i ];
            output[ s_base[ static_cast< int >( radixSortDigit< Value, descending >( value, pass ) ) ] + i ] = value;
         }
      }
      __syncthreads();
   }
   Backend::triggerProgrammaticLaunchCompletion();
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
   Backend::gridDependencySynchronize();
   const Index digit = static_cast< Index >( threadIdx.x );
   const unsigned int word = tileStatus[ ( numTiles - 1 ) * radixSortBuckets + digit ];
   digitBase[ static_cast< Index >( pass ) * radixSortBuckets + digit ] += word & radixSortTileValueMask;
   Backend::triggerProgrammaticLaunchCompletion();
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
   Backend::gridDependencySynchronize();
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
 * of the items follows \ref radixSortOnesweepKernel (warp match + raking
 * scan), except that the match keeps the native intrinsic: the last round
 * of a partial tile runs on a partial warp, which
 * \ref radixSortWarpMatchAny does not support. The permutations of the
 * passes ping-pong between two
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

#if defined( __CUDACC__ ) || defined( __HIP__ )

   //! \brief CUDA/HIP graph type aliases used by \ref RadixSortGraphCache.
   #if defined( __CUDACC__ )
using RadixSortGraphT = cudaGraph_t;
using RadixSortGraphExecT = cudaGraphExec_t;
using RadixSortGraphNodeT = cudaGraphNode_t;
using RadixSortKernelNodeParams = cudaKernelNodeParams;
   #else
using RadixSortGraphT = hipGraph_t;
using RadixSortGraphExecT = hipGraphExec_t;
using RadixSortGraphNodeT = hipGraphNode_t;
using RadixSortKernelNodeParams = hipKernelNodeParams;
   #endif

//! \brief Begins the capture of the launches issued to `stream` (the
//! thread-local mode keeps the capture unaffected by the other threads).
inline void
radixSortGraphBeginCapture( Backend::stream_t stream )
{
   #if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaStreamBeginCapture( stream, cudaStreamCaptureModeThreadLocal ) );
   #else
   TNL_BACKEND_SAFE_CALL( hipStreamBeginCapture( stream, hipStreamCaptureModeThreadLocal ) );
   #endif
}

//! \brief Ends the capture and returns the captured graph template.
inline RadixSortGraphT
radixSortGraphEndCapture( Backend::stream_t stream )
{
   RadixSortGraphT graph;
   #if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaStreamEndCapture( stream, &graph ) );
   #else
   TNL_BACKEND_SAFE_CALL( hipStreamEndCapture( stream, &graph ) );
   #endif
   return graph;
}

/**
 * \brief Returns the handle of the node that the next operation captured by
 * `stream` would depend on.
 *
 * The captured launch sequence is a linear chain, so the single dependency
 * is the handle of the last appended node. The handle belongs to the
 * captured graph template, which must stay alive for the executable-graph
 * parameter updates (the graph API refers the updated nodes by their
 * template handles).
 */
inline RadixSortGraphNodeT
radixSortGraphLastNode( Backend::stream_t stream )
{
   #if defined( __CUDACC__ )
   cudaStreamCaptureStatus status;
   cudaGraph_t graph;
   const cudaGraphNode_t* dependencies = nullptr;
   std::size_t dependenciesCount = 0;
      #if CUDART_VERSION >= 12000
   TNL_BACKEND_SAFE_CALL(
      cudaStreamGetCaptureInfo( stream, &status, nullptr, &graph, &dependencies, nullptr, &dependenciesCount ) );
      #else
   TNL_BACKEND_SAFE_CALL( cudaStreamGetCaptureInfo( stream, &status, nullptr, &graph, &dependencies, &dependenciesCount ) );
      #endif
   #else
   hipStreamCaptureStatus status;
   hipGraph_t graph;
   const hipGraphNode_t* dependencies = nullptr;
   std::size_t dependenciesCount = 0;
   TNL_BACKEND_SAFE_CALL( hipStreamGetCaptureInfo_v2( stream, &status, nullptr, &graph, &dependencies, &dependenciesCount ) );
   #endif
   TNL_ASSERT_EQ( dependenciesCount, 1u, "the captured sequence is a linear chain" );
   return dependencies[ 0 ];
}

//! \brief Instantiates the captured graph template into an executable graph.
inline void
radixSortGraphInstantiate( RadixSortGraphExecT& graphExec, RadixSortGraphT graph )
{
   #if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaGraphInstantiate( &graphExec, graph, 0 ) );
   #else
   TNL_BACKEND_SAFE_CALL( hipGraphInstantiate( &graphExec, graph, nullptr, nullptr, 0 ) );
   #endif
}

//! \brief Launches the executable graph on `stream`.
inline void
radixSortGraphLaunch( RadixSortGraphExecT graphExec, Backend::stream_t stream )
{
   #if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaGraphLaunch( graphExec, stream ) );
   #else
   TNL_BACKEND_SAFE_CALL( hipGraphLaunch( graphExec, stream ) );
   #endif
}

//! \brief Updates the parameters of one kernel node of the executable
//! graph (the node handle refers to the template the executable was
//! instantiated from).
inline void
radixSortGraphExecSetKernelNodeParams(
   RadixSortGraphExecT graphExec,
   RadixSortGraphNodeT node,
   const RadixSortKernelNodeParams& params )
{
   #if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaGraphExecKernelNodeSetParams( graphExec, node, &params ) );
   #else
   TNL_BACKEND_SAFE_CALL( hipGraphExecKernelNodeSetParams( graphExec, node, &params ) );
   #endif
}

//! \brief Destroys the executable graph.
inline void
radixSortGraphExecDestroy( RadixSortGraphExecT& graphExec )
{
   #if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaGraphExecDestroy( graphExec ) );
   #else
   TNL_BACKEND_SAFE_CALL( hipGraphExecDestroy( graphExec ) );
   #endif
   graphExec = nullptr;
}

//! \brief Destroys the graph template.
inline void
radixSortGraphDestroy( RadixSortGraphT& graph )
{
   #if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaGraphDestroy( graph ) );
   #else
   TNL_BACKEND_SAFE_CALL( hipGraphDestroy( graph ) );
   #endif
   graph = nullptr;
}

//! \brief Creates the non-blocking stream used for the graph capture.
inline void
radixSortGraphCreateCaptureStream( Backend::stream_t& stream )
{
   #if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) );
   #else
   TNL_BACKEND_SAFE_CALL( hipStreamCreateWithFlags( &stream, hipStreamNonBlocking ) );
   #endif
}

//! \brief Destroys the capture stream.
inline void
radixSortGraphDestroyStream( Backend::stream_t& stream )
{
   #if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaStreamDestroy( stream ) );
   #else
   TNL_BACKEND_SAFE_CALL( hipStreamDestroy( stream ) );
   #endif
   stream = nullptr;
}

/**
 * \brief Cache of the captured kernel-launch sequence of the radix sort
 * driver.
 *
 * The host-side launch sequence of the radix sort is static for a given
 * (array size, sort order, device, set of workspace buffer pointers): all
 * grid sizes and kernel arguments are computed on the host and only the
 * pass-skip decisions are made on the device. A workspace therefore caches
 * one captured CUDA/HIP graph and replays it with a single graph launch on
 * the calls with a matching key, which removes the per-kernel launch
 * overhead of the driver.
 *
 * The kernels referencing the user array (the total histogram, the binnings
 * and the copy-back) take it as their first argument, so a sort of a
 * *different* array of the cached size and order patches only the data
 * pointer of the recorded kernel arguments (the array views carry the data
 * pointer as their first member).
 *
 * \note The cache is thread-compatible with the workspace contract: the
 * sorts sharing one workspace must be ordered on the same stream, which
 * also orders the replays of the cached graph. One executable graph must
 * not be relaunched while its previous launch is still pending on another
 * stream; the single-stream contract implies that.
 */
struct RadixSortGraphCache
{
   //! Maximum number of kernel arguments of a recorded node (the widest
   //! signature is the binning kernel with 9 arguments).
   static constexpr std::size_t maxNodeArgs = 10;

   //! \brief Recorded kernel node referencing the user array, with owned
   //! copies of its arguments.
   struct ViewNode
   {
      //! Handle of the node in the captured graph template.
      RadixSortGraphNodeT node = nullptr;
      //! Kernel parameters referencing `argStorage`.
      RadixSortKernelNodeParams params{};
      //! Pointers to the arguments stored in `argStorage`.
      void* argPointers[ maxNodeArgs ] = {};
      //! Owned copies of the kernel arguments, 16-byte aligned per
      //! argument; the first argument is always the array view of the
      //! user data, whose data pointer is its first member.
      alignas( 16 ) unsigned char argStorage[ 16 * maxNodeArgs ] = {};
   };

   RadixSortGraphCache() = default;

   //! Destroys the graph objects and the capture stream.
   ~RadixSortGraphCache()
   {
      clearGraph();
      if( captureStream != nullptr )
         radixSortGraphDestroyStream( captureStream );
   }

   // the graph handles are uniquely owned; copying is implemented by the
   // workspace (it copies the buffers and leaves a default-constructed
   // cache which captures its own graph on demand)
   RadixSortGraphCache( const RadixSortGraphCache& ) = delete;
   RadixSortGraphCache&
   operator=( const RadixSortGraphCache& ) = delete;

   RadixSortGraphCache( RadixSortGraphCache&& other ) noexcept
   {
      stealFrom( other );
   }

   RadixSortGraphCache&
   operator=( RadixSortGraphCache&& other ) noexcept
   {
      if( this != &other ) {
         clearGraph();
         if( captureStream != nullptr )
            radixSortGraphDestroyStream( captureStream );
         stealFrom( other );
      }
      return *this;
   }

   //! \brief Destroys the captured graph objects and drops the cache key
   //! (the capture stream is kept for the next capture).
   void
   clearGraph()
   {
      if( graphExec != nullptr )
         radixSortGraphExecDestroy( graphExec );
      if( graph != nullptr )
         radixSortGraphDestroy( graph );
      size = 0;
      descending = false;
      device = -1;
      auxData = nullptr;
      digitBaseData = nullptr;
      tileStatusData = nullptr;
      passFlagsData = nullptr;
      viewData = nullptr;
      viewNodes.clear();
   }

   /**
    * \brief Patches the user-array pointer in the recorded kernel nodes.
    *
    * Every recorded kernel takes the user array view as its first argument
    * and the array view stores the data pointer as its first member, so
    * patching the pointer of the first argument redirects the kernels to
    * the new array (the view size stays the cached size).
    */
   void
   patchViews( const void* newData )
   {
      for( auto& viewNode : viewNodes ) {
         std::memcpy( viewNode.argStorage, &newData, sizeof( newData ) );
         radixSortGraphExecSetKernelNodeParams( graphExec, viewNode.node, viewNode.params );
      }
      viewData = newData;
   }

   //! \brief Stream used for the graph capture (owned, created lazily). The
   //! capture cannot use the driver's stream: the launches issued during
   //! the capture would otherwise become an implicit part of the graph of
   //! the caller instead of being replayed on it.
   Backend::stream_t captureStream = nullptr;

   //! \brief Captured launch sequence; the node handles of `viewNodes`
   //! refer to this template, which must therefore stay alive.
   RadixSortGraphT graph = nullptr;

   //! \brief Executable graph replayed by the sorts with a matching key.
   RadixSortGraphExecT graphExec = nullptr;

   //! \brief Array size the graph was captured for (the launch
   //! configuration depends on it).
   std::size_t size = 0;

   //! \brief Sort order the graph was captured for (a workspace shared by
   //! ascending and descending sorts).
   bool descending = false;

   //! \brief Device the graph was captured on (-1 while no graph is held).
   int device = -1;

   //! \brief Buffer pointers referenced by the captured kernels; a change
   //! (e.g., after the workspace grows) invalidates the graph.
   const void* auxData = nullptr;
   const void* digitBaseData = nullptr;
   const void* tileStatusData = nullptr;
   const void* passFlagsData = nullptr;

   //! \brief User-array pointer referenced by the recorded kernel nodes.
   const void* viewData = nullptr;

   //! \brief Recorded kernel nodes referencing the user array (the total
   //! histogram, the binnings and the copy-back). A deque keeps the node
   //! objects address-stable, which the parameters' pointers into
   //! `argStorage` rely on.
   std::deque< ViewNode > viewNodes;

private:
   //! \brief Takes the ownership of \p other (leaving it empty).
   void
   stealFrom( RadixSortGraphCache& other ) noexcept
   {
      captureStream = other.captureStream;
      graph = other.graph;
      graphExec = other.graphExec;
      size = other.size;
      descending = other.descending;
      device = other.device;
      auxData = other.auxData;
      digitBaseData = other.digitBaseData;
      tileStatusData = other.tileStatusData;
      passFlagsData = other.passFlagsData;
      viewData = other.viewData;
      viewNodes = std::move( other.viewNodes );
      other.captureStream = nullptr;
      other.graph = nullptr;
      other.graphExec = nullptr;
      other.size = 0;
      other.device = -1;
      other.viewData = nullptr;
   }
};

/**
 * \brief Records the kernel that was just launched into the capturing
 * stream as a view-referencing node of the cache.
 *
 * The arguments are copied into the node record so that the user-array
 * pointer can be patched later (see \ref RadixSortGraphCache); the argument
 * values must be identical to the ones of the launch.
 */
template< typename KernelFunction, typename... Args >
void
radixSortRecordGraphViewNode(
   RadixSortGraphCache& cache,
   Backend::stream_t stream,
   KernelFunction kernel,
   const Backend::LaunchConfiguration& launch_config,
   const Args&... args )
{
   static_assert( sizeof...( Args ) <= RadixSortGraphCache::maxNodeArgs );
   // the record is built in place: the kernel parameters point into the
   // node's own storage, which must stay address-stable (see the deque of
   // RadixSortGraphCache)
   auto& viewNode = cache.viewNodes.emplace_back();
   std::size_t offset = 0;
   std::size_t argIndex = 0;
   const auto storeArg = [ &viewNode, &offset, &argIndex ]( const auto& arg )
   {
      // the kernel arguments are plain aggregates (pointers, sizes and the
      // (pointer, size) array views), so they are copied as bytes; note
      // that the views are not "trivially copyable" in the sense of the
      // type traits when compiled with -rdc=true, but they stay
      // standard-layout aggregates here
      static_assert( std::is_standard_layout_v< std::decay_t< decltype( arg ) > > );
      std::memcpy( viewNode.argStorage + offset, &arg, sizeof( arg ) );
      viewNode.argPointers[ argIndex ] = viewNode.argStorage + offset;
      argIndex++;
      offset += ( sizeof( arg ) + 15 ) / 16 * 16;
   };
   ( storeArg( args ), ... );
   TNL_ASSERT_LE( offset, sizeof( viewNode.argStorage ), "the recorded arguments must fit the node storage" );
   viewNode.params.func = reinterpret_cast< void* >( kernel );
   viewNode.params.gridDim = launch_config.gridSize;
   viewNode.params.blockDim = launch_config.blockSize;
   viewNode.params.sharedMemBytes = static_cast< unsigned int >( launch_config.dynamicSharedMemorySize );
   viewNode.params.kernelParams = viewNode.argPointers;
   viewNode.params.extra = nullptr;
   viewNode.node = radixSortGraphLastNode( stream );
}

#endif

/**
 * \brief Workspace of the temporary buffers of the radix sort driver.
 *
 * The radix sort driver needs an auxiliary array of the size of the input
 * (the ping-pong buffer of the sweeps), the digit histograms of the passes,
 * the tile status counters of the decoupled lookback and the pass flags.
 * The latter three metadata buffers are views into a single allocation to
 * reduce the host-side allocation cost of a fresh workspace.
 * A default-constructed workspace owns no memory; \ref setSize grows the
 * buffers to the requested size and they are never shrunk, so a workspace
 * reused across sorts performs no device memory allocation at all once the
 * buffers fit the input. The auxiliary array is the dominant buffer (one
 * value per element), but even the small metadata allocations cost host
 * time when a fresh workspace is constructed and destroyed for every sort.
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

   RadixSortWorkspace() = default;
   ~RadixSortWorkspace() = default;

   /**
    * \brief Copying a workspace copies the buffers; the cached graph is
    * not copied (the copy captures its own graph on the first use).
    */
   RadixSortWorkspace( const RadixSortWorkspace& other )
   : aux( other.aux ),
     smallBuffers( other.smallBuffers )
   {
      bindSmallBuffers( other.tileStatus.getSize() );
   }

   RadixSortWorkspace&
   operator=( const RadixSortWorkspace& other )
   {
      const Index tileStatusSize = other.tileStatus.getSize();
      aux = other.aux;
      smallBuffers = other.smallBuffers;
      bindSmallBuffers( tileStatusSize );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      // the buffers were deep-copied, so a cached graph of the assigned
      // workspace must not be replayed against them
      graphCache.clearGraph();
#endif
      return *this;
   }

   RadixSortWorkspace( RadixSortWorkspace&& other ) noexcept
   : aux( std::move( other.aux ) ),
     smallBuffers( std::move( other.smallBuffers ) )
#if defined( __CUDACC__ ) || defined( __HIP__ )
     ,
     graphCache( std::move( other.graphCache ) )
#endif
   {
      bindSmallBuffers( other.tileStatus.getSize() );
      other.resetSmallBufferViews();
   }

   RadixSortWorkspace&
   operator=( RadixSortWorkspace&& other ) noexcept( false )
   {
      if( this != &other ) {
         const Index tileStatusSize = other.tileStatus.getSize();
         aux = std::move( other.aux );
         smallBuffers = std::move( other.smallBuffers );
#if defined( __CUDACC__ ) || defined( __HIP__ )
         graphCache = std::move( other.graphCache );
#endif
         bindSmallBuffers( tileStatusSize );
         other.resetSmallBufferViews();
      }
      return *this;
   }

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

      constexpr int passes = static_cast< int >( sizeof( Value ) );
      Index requiredTileStatusSize = tileStatus.getSize();

      // one status counter per (tile, digit) plus the work counter, with
      // one such region per radix pass; the number of tiles of a strip is
      // capped by the maximum strip size of 2^28 elements, so the buffer
      // stops growing once it can hold the counters of a full strip
      if( size > 0 ) {
         const Index tileSize = radixSortOnesweepTileSizeFor< Value >( size );
         // computed without rounding up to avoid overflow of the index
         // type when the size is close to its maximum; the tiles of a full
         // strip are rounded up (a partial tile of a strip gets its own
         // status counters, plus one work counter at the end)
         const Index numTiles = size / tileSize + static_cast< Index >( size % tileSize != 0 );
         const Index maxStripTiles = TNL::min( numTiles, ( ( static_cast< Index >( 1 ) << 28 ) + tileSize - 1 ) / tileSize );
         // padded to a multiple of the bucket count so that every region
         // starts 1 KiB-aligned (keep in sync with the driver in radixSort)
         const Index requiredSize = static_cast< Index >( passes ) * ( ( maxStripTiles + 1 ) * radixSortBuckets );
         requiredTileStatusSize = TNL::max( requiredTileStatusSize, requiredSize );
      }

      // the histogram and flag sizes depend only on the value type; their
      // fixed offsets leave the variable-sized tile status region at the end
      if( smallBuffers.getSize() == 0 || tileStatus.getSize() < requiredTileStatusSize ) {
         smallBuffers.setSize( getSmallBufferWords( requiredTileStatusSize ) );
         bindSmallBuffers( requiredTileStatusSize );
      }
   }

   //! \brief Auxiliary (ping-pong) array of at least the sorted size.
   Containers::Array< Value, Device, Index > aux;

private:
   // 64-bit words align all supported index types; 1 KiB-relative region
   // offsets preserve the tile status alignment assumed by the driver
   static constexpr std::size_t smallBufferAlignment = 1024;

   static constexpr std::size_t
   alignSmallBufferOffset( const std::size_t offset )
   {
      return ( offset + smallBufferAlignment - 1 ) / smallBufferAlignment * smallBufferAlignment;
   }

   static constexpr std::size_t digitBaseBytes = sizeof( Value ) * radixSortBuckets * sizeof( Index );
   static constexpr std::size_t passFlagsOffset = alignSmallBufferOffset( digitBaseBytes );
   static constexpr std::size_t passFlagsBytes = ( 2 * sizeof( Value ) + 1 ) * sizeof( int );
   static constexpr std::size_t tileStatusOffset = alignSmallBufferOffset( passFlagsOffset + passFlagsBytes );

   static Index
   getSmallBufferWords( const Index tileStatusSize )
   {
      const std::size_t bytes = tileStatusOffset + static_cast< std::size_t >( tileStatusSize ) * sizeof( unsigned int );
      return static_cast< Index >( ( bytes + sizeof( std::uint64_t ) - 1 ) / sizeof( std::uint64_t ) );
   }

   //! \brief Owning allocation shared by the three metadata views.
   Containers::Array< std::uint64_t, Device, Index > smallBuffers;

   void
   bindSmallBuffers( const Index tileStatusSize )
   {
      if( smallBuffers.getSize() == 0 ) {
         resetSmallBufferViews();
         return;
      }
      auto* const base = reinterpret_cast< std::uint8_t* >( smallBuffers.getData() );
      constexpr int passes = static_cast< int >( sizeof( Value ) );
      digitBase.bind( reinterpret_cast< Index* >( base ), passes * radixSortBuckets );
      passFlags.bind( reinterpret_cast< int* >( base + passFlagsOffset ), 2 * passes + 1 );
      if( tileStatusSize > 0 )
         tileStatus.bind( reinterpret_cast< unsigned int* >( base + tileStatusOffset ), tileStatusSize );
      else
         tileStatus.reset();
   }

   void
   resetSmallBufferViews()
   {
      digitBase.reset();
      tileStatus.reset();
      passFlags.reset();
   }

public:
   //! \brief Base offsets of the digit runs: `passes` x `radixSortBuckets` entries.
   Containers::ArrayView< Index, Device, Index > digitBase;

   //! \brief Status counters of the lookback tiles plus the work counter,
   //! with one region per radix pass (see \ref radixSort).
   Containers::ArrayView< unsigned int, Device, Index > tileStatus;

   //! \brief Skip flags and buffer-parity flags of the passes: `2 * passes + 1` entries.
   Containers::ArrayView< int, Device, Index > passFlags;

#if defined( __CUDACC__ ) || defined( __HIP__ )
   /**
    * \brief Cache of the captured kernel-launch sequence replayed by the
    * sorts of a matching (array size, sort order, device and buffer
    * pointers), removing the per-kernel launch overhead of the driver.
    */
   RadixSortGraphCache graphCache;
#endif
};

/**
 * \brief Launches the kernel sequence of the radix sort driver on the
 * given stream.
 *
 * The sequence is host-static for a given (array size, sort order, device
 * and set of workspace buffer pointers): all kernels are launched
 * unconditionally with host-computed launch configurations and the
 * pass-skip decisions are read by the sweeps on the device (see \ref
 * radixSortProcessHistogramsKernel). When \p cache is not null, the stream
 * must be under graph capture and the sequence additionally records the
 * kernels referencing the user array view into the cache; the captured
 * kernels do not execute, so the caller replays the instantiated graph to
 * perform the sort.
 */
template< typename Value, typename Device, typename Index, bool descending >
void
radixSortLaunchSequence(
   Containers::ArrayView< Value, Device, Index > view,
   RadixSortWorkspace< Value, Device, Index >& workspace,
   const Index n,
   Backend::stream_t stream,
   RadixSortGraphCache* cache )
{
   constexpr int passes = static_cast< int >( sizeof( Value ) );
   const bool usePdl = n <= static_cast< Index >( radixSortPdlLimit );

   // computed without rounding up to avoid overflow of n + tileSize when
   // the index type is signed and n is close to its maximum; the small
   // arrays take the short-tile configuration of the sweep (see
   // radixSortOnesweepSmallTileLimit)
   const Index tileSize = radixSortOnesweepTileSizeFor< Value >( n );
   const Index numTiles = n / tileSize + static_cast< Index >( n % tileSize != 0 );
   const int multiprocessors = Backend::getDeviceMultiprocessors( Backend::getDevice() );
   const Index maxBlocks = static_cast< Index >( multiprocessors ) * 32;

   // the sweeps of the executed passes alternate between view and the
   // auxiliary array of the workspace
   auto aux = workspace.aux.getView( 0, n );

   Backend::LaunchConfiguration launch_config;
   launch_config.programmaticStreamSerialization = usePdl;
   launch_config.stream = stream;
   launch_config.blockSize.x = radixSortBlockSize;
   launch_config.gridSize.x = static_cast< unsigned >( TNL::min( numTiles, maxBlocks ) );

   // the sweep kernel decouples its block size from the bucket count (the
   // per-digit phases are guarded inside the kernel); the dispatch is a pure
   // function of the array size and value type, so it is also graph-capturable
   Backend::LaunchConfiguration onesweepConfig = launch_config;
   constexpr auto onesweepKernelSmall =
      radixSortOnesweepKernel< Value, Index, descending, radixSortBlockSize, radixSortItemsPerThread >;
   constexpr auto onesweepKernelMedium = radixSortOnesweepKernel<
      Value,
      Index,
      descending,
      radixSortOnesweepMediumBlockSize,
      radixSortOnesweepMediumItemsPerThread >;
   constexpr auto onesweepKernelBig = radixSortOnesweepKernel< Value, Index, descending >;
   auto onesweepKernel = onesweepKernelBig;
   onesweepConfig.blockSize.x = radixSortOnesweepBlockSize;
   if( n <= static_cast< Index >( radixSortOnesweepSmallTileLimit< Value > ) ) {
      onesweepKernel = onesweepKernelSmall;
      onesweepConfig.blockSize.x = radixSortBlockSize;
   }
   else if( n <= static_cast< Index >( radixSortOnesweepMediumTileLimit ) ) {
      onesweepKernel = onesweepKernelMedium;
      onesweepConfig.blockSize.x = radixSortOnesweepMediumBlockSize;
   }

   // launch configuration for zero-filling the work buffers: the fills are
   // ordered with the sort kernels on the stream and do not block the
   // host, which keeps them valid also inside the graph capture
   Devices::GPU::LaunchConfiguration fillConfig;
   fillConfig.stream = stream;
   fillConfig.blockHostUntilFinished = false;

   // the digit counts are independent of the order of the elements, which
   // changes between the passes, so the histograms of all passes are
   // computed upfront; the exclusive scan of each histogram then gives the
   // global base offsets of the digit runs
   auto& digitBase = workspace.digitBase;
   auto& passFlags = workspace.passFlags;
   Algorithms::fill< Devices::GPU >( digitBase.getData(), Index( 0 ), digitBase.getSize(), fillConfig );
   // The first fill serializes with the input producer; all later kernels use
   // explicit grid dependency synchronization and may be dependent launches.
   fillConfig.programmaticStreamSerialization = usePdl;

   constexpr auto totalHistogramKernel = radixSortTotalHistogramKernel< Value, Index, descending >;
   Backend::launchKernelAsyncWithAttributes(
      totalHistogramKernel, launch_config, view, digitBase.getView(), passFlags.getView() );
   if( cache != nullptr )
      radixSortRecordGraphViewNode(
         *cache, stream, totalHistogramKernel, launch_config, view, digitBase.getView(), passFlags.getView() );

   // scan the digit histograms of all passes and detect the skippable
   // passes together with the source/target buffer parity of each sweep in
   // a single kernel, so that the host can launch the sweeps of all passes
   // without knowing which of them are going to run; the fused kernel
   // replaces the per-pass non-blocking launches of the standalone
   // Algorithms::inplaceExclusiveScan
   constexpr auto processHistogramsKernel = radixSortProcessHistogramsKernel< Value, Index >;
   Backend::LaunchConfiguration processHistogramsConfig( static_cast< unsigned >( passes ), radixSortBuckets );
   processHistogramsConfig.programmaticStreamSerialization = usePdl;
   processHistogramsConfig.stream = stream;
   Backend::launchKernelAsyncWithAttributes(
      processHistogramsKernel, processHistogramsConfig, digitBase.getView(), passFlags.getView() );

   // the input is processed in strips of up to 2^28 elements: the values
   // exchanged by the decoupled lookback must fit into 30 bits, and the
   // tile status array (half a byte per element, one counter per tile and
   // digit) stays reasonably small
   constexpr Index onesweepStripSize = ( static_cast< Index >( 1 ) << 28 );
   const Index numStrips = n / onesweepStripSize + static_cast< Index >( n % onesweepStripSize != 0 );
   const Index maxStripTiles = TNL::min( numTiles, ( onesweepStripSize + tileSize - 1 ) / tileSize );

   // each pass owns one region of the tile status buffer, holding one
   // status counter per (tile, digit) of a strip plus the work counter;
   // the region size is padded to a multiple of the bucket count, so every
   // region starts on the same 1 KiB alignment as the single-region layout
   // and the per-tile counter bursts stay line-aligned in all passes;
   // the buffer is reused, so it may be larger than needed, but the kernels
   // of this call touch only the used prefix of the buffer
   auto& tileStatus = workspace.tileStatus;
   const Index tileStatusSize = ( maxStripTiles + 1 ) * radixSortBuckets;

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
         const Index stripNumTiles = stripSize / tileSize + static_cast< Index >( stripSize % tileSize != 0 );

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
         onesweepConfig.gridSize.x = static_cast< unsigned >( TNL::min( stripNumTiles, maxBlocks ) );
         Backend::launchKernelAsyncWithAttributes(
            onesweepKernel,
            onesweepConfig,
            view,
            aux,
            digitBase.getView(),
            tileStatusRegion,
            passFlags.getConstView(),
            stripBegin,
            stripEnd,
            stripNumTiles,
            pass );
         if( cache != nullptr )
            radixSortRecordGraphViewNode(
               *cache,
               stream,
               onesweepKernel,
               onesweepConfig,
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
            Backend::launchKernelAsyncWithAttributes(
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
   Backend::launchKernelAsyncWithAttributes( copyBackKernel, launch_config, view, aux, passFlags.getConstView() );
   if( cache != nullptr )
      radixSortRecordGraphViewNode( *cache, stream, copyBackKernel, launch_config, view, aux, passFlags.getConstView() );
}

/**
 * \brief Implementation of \ref radixSort with an optional use of the
 * graph cache of the workspace.
 */
template< typename Value, typename Device, typename Index, bool descending >
void
radixSortImpl(
   Containers::ArrayView< Value, Device, Index > view,
   RadixSortWorkspace< Value, Device, Index >& workspace,
   const bool useGraphCache )
{
   static_assert( std::is_same_v< Device, Devices::GPU >, "RadixSort is implemented only for the GPU device" );
   static_assert( std::is_arithmetic_v< Value >, "RadixSort supports only arithmetic value types" );
   static_assert( sizeof( Value ) <= 8, "RadixSort supports value types with at most 8 bytes" );

#if defined( __CUDACC__ ) || defined( __HIP__ )
   // early exit for empty or single-element arrays (already sorted)
   if( view.getSize() <= 1 )
      return;

   const Index n = view.getSize();

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

   if( useGraphCache && n <= static_cast< Index >( radixSortGraphLimit ) ) {
      auto& cache = workspace.graphCache;
      // an executable graph is replayed only when everything that the
      // captured kernels refer to is unchanged: the array size (the launch
      // configuration), the sort order, the device, and the pointers of
      // the workspace buffers (a growth of the workspace reallocates them)
      const bool ready = cache.graphExec != nullptr && cache.size == static_cast< std::size_t >( n )
                      && cache.descending == descending && cache.device == Backend::getDevice()
                      && cache.auxData == static_cast< const void* >( workspace.aux.getData() )
                      && cache.digitBaseData == static_cast< const void* >( workspace.digitBase.getData() )
                      && cache.tileStatusData == static_cast< const void* >( workspace.tileStatus.getData() )
                      && cache.passFlagsData == static_cast< const void* >( workspace.passFlags.getData() );
      if( ready ) {
         // the kernels reference the user array by their first argument:
         // a sort of a different array of the cached size and order only
         // patches the recorded pointers, which costs a fraction of a
         // microsecond per node
         if( cache.viewData != static_cast< const void* >( view.getData() ) )
            cache.patchViews( view.getData() );
         radixSortGraphLaunch( cache.graphExec, 0 );
         Backend::streamSynchronize( 0 );
         return;
      }

      // no usable executable graph: drop the stale state and capture the
      // launch sequence on the internal stream; the captured kernels do
      // not execute, so the cold call ends by replaying the fresh
      // executable graph on the driver's stream
      cache.clearGraph();
      if( cache.captureStream == nullptr )
         radixSortGraphCreateCaptureStream( cache.captureStream );
      radixSortGraphBeginCapture( cache.captureStream );
      radixSortLaunchSequence< Value, Device, Index, descending >( view, workspace, n, cache.captureStream, &cache );
      cache.graph = radixSortGraphEndCapture( cache.captureStream );
      radixSortGraphInstantiate( cache.graphExec, cache.graph );
      cache.size = static_cast< std::size_t >( n );
      cache.descending = descending;
      cache.device = Backend::getDevice();
      cache.auxData = workspace.aux.getData();
      cache.digitBaseData = workspace.digitBase.getData();
      cache.tileStatusData = workspace.tileStatus.getData();
      cache.passFlagsData = workspace.passFlags.getData();
      cache.viewData = view.getData();
      radixSortGraphLaunch( cache.graphExec, 0 );
      Backend::streamSynchronize( 0 );
      return;
   }

   radixSortLaunchSequence< Value, Device, Index, descending >( view, workspace, n, 0, nullptr );
   Backend::streamSynchronize( 0 );
#else
   (void) useGraphCache;
   throw Exceptions::NotImplementedError( "RadixSort is supported only when CUDA or HIP is enabled." );
#endif
}

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
 * The launch sequence of a sort is host-static for a given (array size,
 * sort order, device and set of workspace buffer pointers): the driver
 * captures it into an executable CUDA/HIP graph on the first call with a
 * given workspace and the calls with a matching key replay it with a
 * single graph launch, which removes most of the host-side launch overhead
 * of the driver. Sorting a different array of the cached size and order
 * reuses the graph as well: the kernels take the array as their first
 * argument, so only the recorded pointer is patched before the replay. A
 * sort with a different size or order recaptures the sequence in its
 * workspace (a handful of microseconds). The replay is limited to arrays
 * of at most \ref radixSortGraphLimit elements: longer arrays keep the
 * direct launches, which hide their own overhead once their kernels are
 * long enough to keep the device busy while the host issues the sequence
 * (a replayed graph would still pay the device-side node scheduling).
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
   radixSortImpl< Value, Device, Index, descending >( view, workspace, true );
}

/**
 * \brief One-shot overload of \ref radixSort which allocates a fresh
 * workspace per call.
 *
 * Repeated sorts should prefer the overload taking \ref RadixSortWorkspace,
 * which removes the per-call device memory allocations.
 *
 * Unlike the workspace overload, the launch sequence is never captured
 * into an executable graph: the fresh workspace would have to recapture it
 * on every call, which costs more than the kernel launches it saves.
 */
template< typename Value, typename Device, typename Index, bool descending >
void
radixSort( Containers::ArrayView< Value, Device, Index > view )
{
   RadixSortWorkspace< Value, Device, Index > workspace;
   radixSortImpl< Value, Device, Index, descending >( view, workspace, false );
}

}  // namespace TNL::Algorithms::Sorting::detail
