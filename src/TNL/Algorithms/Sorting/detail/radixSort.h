// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

#include <TNL/Backend.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL::Algorithms::Sorting::detail {

//! \brief Number of bits of the key processed in one radix sort pass.
constexpr int radixSortBits = 8;

//! \brief Number of digit buckets, i.e. `2^radixSortBits`.
constexpr int radixSortBuckets = 1 << radixSortBits;

//! \brief Number of threads per block, which is also the number of elements
//! processed by one tile of the kernels.
constexpr int radixSortTileSize = 256;

// The scatter kernel relies on each thread handling exactly one element per
// tile and one digit in the per-digit phases, which requires the tile size
// to be equal to the number of digit buckets.
static_assert( radixSortTileSize == radixSortBuckets, "radixSortTileSize must be equal to radixSortBuckets" );

//! \brief Maximum number of warps per block in the radix sort kernels,
//! assuming the smallest supported warp size of 32.
constexpr int radixSortMaxWarps = radixSortTileSize / 32;

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

/**
 * \brief Kernel computing per-block histograms of digit counts for one pass.
 *
 * The input is partitioned into tiles of `radixSortTileSize` elements and
 * each block of the grid processes a contiguous span of tiles. The counts
 * are accumulated in shared memory and written to `histograms` in the
 * digit-major layout, i.e. the entry for the digit `d` and block `b` is
 * stored at `histograms[ d * gridDim.x + b ]`.
 *
 * After an exclusive scan of the whole `histograms` array (done by the host
 * code between the two kernels), the entry for the digit `d` and block `b`
 * is the offset where the elements of the block `b` with digit `d` shall
 * be scattered in this pass. The digit-major layout is essential: it makes
 * the scanned offsets order the elements by the digit first, which is
 * required for the correctness of LSD radix sort.
 *
 * \param input is the view of the array to be sorted.
 * \param histograms is the view of per-block histograms to be filled.
 * \param numTiles is the total number of tiles of the input.
 * \param pass is the index of the radix sort pass.
 */
template< typename Value, typename Index, bool descending >
__global__
void
radixSortHistogramKernel(
   const Containers::ArrayView< Value, Devices::GPU, Index > input,
   Containers::ArrayView< Index, Devices::GPU, Index > histograms,
   Index numTiles,
   int pass )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ unsigned int s_counts[ radixSortBuckets ];

   for( int d = threadIdx.x; d < radixSortBuckets; d += blockDim.x )
      s_counts[ d ] = 0;
   __syncthreads();

   const Index n = input.getSize();
   const Index tilesPerBlock = ( numTiles + static_cast< Index >( gridDim.x ) - 1 ) / static_cast< Index >( gridDim.x );
   const Index firstTile = static_cast< Index >( blockIdx.x ) * tilesPerBlock;
   const Index lastTile = TNL::min( numTiles, firstTile + tilesPerBlock );

   for( Index tile = firstTile; tile < lastTile; tile++ ) {
      const Index tileBegin = tile * static_cast< Index >( radixSortTileSize );
      const Index tileEnd = tileBegin + TNL::min( n - tileBegin, static_cast< Index >( radixSortTileSize ) );
      const int tileSize = static_cast< int >( tileEnd - tileBegin );
      // the tile-local loop counter avoids overflow of the index type
      // when the array size is close to its maximum
      for( int i = threadIdx.x; i < tileSize; i += blockDim.x ) {
         const unsigned digit = radixSortDigit< Value, descending >( input[ tileBegin + i ], pass );
         atomicAdd( &s_counts[ digit ], 1u );
      }
   }
   __syncthreads();

   for( int d = threadIdx.x; d < radixSortBuckets; d += blockDim.x )
      histograms[ d * static_cast< Index >( gridDim.x ) + static_cast< Index >( blockIdx.x ) ] =
         static_cast< Index >( s_counts[ d ] );
#endif
}

#if defined( __CUDACC__ ) || defined( __HIP__ )

//! \brief Type of the warp synchronization mask, following
//! \ref Backend::getWarpFullMask (32-bit on CUDA, 64-bit on HIP).
//!
//! \note The warp match intrinsic used by the scatter kernel requires
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

/**
 * \brief Kernel scattering elements to their output positions in one pass.
 *
 * The block-to-tiles mapping is identical to \ref radixSortHistogramKernel,
 * which is essential for correctness. The intra-tile rank of an element
 * among the elements with the same digit is computed with the warp match
 * intrinsic, which is O(1) in the tile size instead of a linear scan over
 * the digits. The shared arrays hold the base offsets `s_base` of this
 * block, advanced by the tile totals after each tile so that they track
 * the current position for each digit, and the per-tile counts `s_warpCounts`
 * of the elements with the same digit per warp, scanned into exclusive
 * prefixes over the warps. The launch uses `blockSize.x` equal to both
 * `radixSortTileSize` and `radixSortBuckets`, so each thread handles
 * exactly one element per tile and one digit in the per-digit phases.
 * The position of an element is composed as
 *
 * \f[ \mathrm{pos} = \mathrm{base}[ d ] + \mathrm{warpPrefix}[ d, w ] + \mathrm{rank}, \f]
 *
 * where `d` is the element's digit, `w` is its warp and `rank` is the number
 * of preceding lanes of the warp with the same digit. Since all parts are
 * computed deterministically, the pass is stable, which is required for the
 * correctness of LSD radix sort.
 *
 * \param input is the view of the array to be sorted.
 * \param output is the view of the auxiliary array for the scattered elements.
 * \param histograms is the view of the exclusive-scanned per-block histograms.
 * \param numTiles is the total number of tiles of the input.
 * \param pass is the index of the radix sort pass.
 */
template< typename Value, typename Index, bool descending >
__global__
void
radixSortScatterKernel(
   const Containers::ArrayView< Value, Devices::GPU, Index > input,
   Containers::ArrayView< Value, Devices::GPU, Index > output,
   const Containers::ArrayView< Index, Devices::GPU, Index > histograms,
   Index numTiles,
   int pass )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ Index s_base[ radixSortBuckets ];
   __shared__ unsigned char s_warpCounts[ radixSortBuckets * radixSortMaxWarps ];

   // load the scanned base offsets of this block's digit counts; they are
   // advanced by the tile totals below, so they hold the current offsets of
   // the elements with each digit already processed by this block
   // (the launch uses blockSize.x == radixSortBuckets, so each thread
   // handles exactly one digit in the per-digit phases below)
   s_base[ threadIdx.x ] = histograms[ threadIdx.x * static_cast< Index >( gridDim.x ) + static_cast< Index >( blockIdx.x ) ];
   __syncthreads();

   const Index n = input.getSize();
   const Index tilesPerBlock = ( numTiles + static_cast< Index >( gridDim.x ) - 1 ) / static_cast< Index >( gridDim.x );
   const Index firstTile = static_cast< Index >( blockIdx.x ) * tilesPerBlock;
   const Index lastTile = TNL::min( numTiles, firstTile + tilesPerBlock );

   const int warpId = static_cast< int >( threadIdx.x ) / warpSize;
   const int lane = static_cast< int >( threadIdx.x ) % warpSize;

   for( Index tile = firstTile; tile < lastTile; tile++ ) {
      const Index tileBegin = tile * static_cast< Index >( radixSortTileSize );
      const Index tileEnd = tileBegin + TNL::min( n - tileBegin, static_cast< Index >( radixSortTileSize ) );
      const int tileSize = static_cast< int >( tileEnd - tileBegin );

      // zero the per-(digit, warp) counts of this tile
      for( int i = threadIdx.x; i < radixSortBuckets * radixSortMaxWarps; i += blockDim.x )
         s_warpCounts[ i ] = 0;
      __syncthreads();

      // each thread handles the element of the tile with its index
      // (the launch uses blockSize.x == radixSortTileSize)
      const bool active = static_cast< int >( threadIdx.x ) < tileSize;
      // active threads of a warp form a prefix of the warp, so the mask of
      // lanes participating in the warp match can be computed directly
      const int activeInWarp =
         TNL::max( 0, TNL::min( static_cast< int >( warpSize ), tileSize - warpId * static_cast< int >( warpSize ) ) );
      const RadixSortWarpMask activeMask =
         activeInWarp >= static_cast< int >( warpSize )
            ? Backend::getWarpFullMask()
            : ( static_cast< RadixSortWarpMask >( 1 ) << activeInWarp ) - static_cast< RadixSortWarpMask >( 1 );
      Value myValue{};
      unsigned myDigit = 0;
      unsigned warpRank = 0;
      if( active ) {
         myValue = input[ tileBegin + threadIdx.x ];
         myDigit = radixSortDigit< Value, descending >( myValue, pass );
         const RadixSortWarpMask match = __match_any_sync( activeMask, myDigit );
         const RadixSortWarpMask laneMaskLt =
            ( static_cast< RadixSortWarpMask >( 1 ) << lane ) - static_cast< RadixSortWarpMask >( 1 );
         warpRank = radixSortPopcount( match & laneMaskLt );
         // the first lane of each group of equal digits stores its count
         if( warpRank == 0 )
            s_warpCounts[ myDigit * radixSortMaxWarps + warpId ] = radixSortPopcount( match );
      }
      __syncthreads();

      // exclusive scan of the warp counts of each digit (in place);
      // the digit total stays in a register for the offset update below
      unsigned short digitTotal = 0;
      for( int w = 0; w < radixSortMaxWarps; w++ ) {
         const auto count = s_warpCounts[ threadIdx.x * radixSortMaxWarps + w ];
         s_warpCounts[ threadIdx.x * radixSortMaxWarps + w ] = static_cast< unsigned char >( digitTotal );
         digitTotal += count;
      }
      __syncthreads();

      if( active ) {
         const Index pos = s_base[ myDigit ] + s_warpCounts[ myDigit * radixSortMaxWarps + warpId ] + warpRank;
         output[ pos ] = myValue;
      }
      __syncthreads();

      // advance the digit offsets by the totals of this tile
      s_base[ threadIdx.x ] += digitTotal;
      __syncthreads();
   }
#endif
}

/**
 * \brief Sorts the given view using LSD radix sort.
 *
 * The algorithm processes the bytes of the order-preserving sort key
 * (see \ref valueToRadixSortKey) from the least significant to the most
 * significant one. Each pass consists of computing per-block histograms of
 * digit counts, exclusive-scanning them, and scattering the elements to
 * an auxiliary array, which is swapped with the input between the passes.
 * All passes are stable, so the overall sort is stable.
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
   const Index numBlocks = TNL::min( numTiles, maxBlocks );

   Containers::Array< Value, Device, Index > aux( n );
   Containers::Array< Index, Device, Index > histograms( numBlocks * radixSortBuckets );

   auto input = view;
   auto output = aux.getView();

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = radixSortTileSize;
   launch_config.gridSize.x = static_cast< unsigned >( numBlocks );

   constexpr auto histogramKernel = radixSortHistogramKernel< Value, Index, descending >;
   constexpr auto scatterKernel = radixSortScatterKernel< Value, Index, descending >;

   // all kernels are launched on the same stream, so their execution is
   // ordered by the stream (the generic scan host-synchronizes internally,
   // so the passes do not actually overlap)
   for( int pass = 0; pass < passes; pass++ ) {
      Backend::launchKernelAsync( histogramKernel, launch_config, input, histograms.getView(), numTiles, pass );

      // exclusive scan of the per-block histograms turns the counts into
      // the base offsets for the scatter kernel
      Algorithms::inplaceExclusiveScan( histograms );

      Backend::launchKernelAsync( scatterKernel, launch_config, input, output, histograms.getView(), numTiles, pass );
      input.swap( output );
   }
   Backend::streamSynchronize( launch_config.stream );

   // for an odd number of passes the sorted data ends up in the auxiliary array
   if( passes % 2 == 1 )
      view = input;
#else
   throw Exceptions::NotImplementedError( "RadixSort is supported only when CUDA or HIP is enabled." );
#endif
}

}  // namespace TNL::Algorithms::Sorting::detail
