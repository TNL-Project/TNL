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
   auto key = valueToRadixSortKey( value );
   if constexpr( descending )
      key = ~key;
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
      const Index end = TNL::min( n, ( tile + 1 ) * static_cast< Index >( radixSortTileSize ) );
      for( Index i = tile * radixSortTileSize + threadIdx.x; i < end; i += blockDim.x ) {
         const unsigned digit = radixSortDigit< Value, descending >( input[ i ], pass );
         atomicAdd( &s_counts[ digit ], 1u );
      }
   }
   __syncthreads();

   for( int d = threadIdx.x; d < radixSortBuckets; d += blockDim.x )
      histograms[ d * static_cast< Index >( gridDim.x ) + static_cast< Index >( blockIdx.x ) ] =
         static_cast< Index >( s_counts[ d ] );
#endif
}

/**
 * \brief Kernel scattering elements to their output positions in one pass.
 *
 * The block-to-tiles mapping is identical to \ref radixSortHistogramKernel,
 * which is essential for correctness. The shared array `s_base` holds the
 * (scanned) base offsets of this block, `s_offset` accumulates the counts
 * of the elements with the same digit already processed by this block,
 * `s_counts` holds the per-tile digit counts, and `s_digits` stores the
 * digits of the current tile. The position of an element is composed as
 *
 * \f[ \mathrm{pos} = \mathrm{base}[ d ] + \mathrm{offset}[ d ] + \mathrm{rank}, \f]
 *
 * where `d` is the element's digit and `rank` is the number of preceding
 * elements in the tile with the same digit. Since all three parts are
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
   __shared__ Index s_offset[ radixSortBuckets ];
   __shared__ unsigned int s_counts[ radixSortBuckets ];
   __shared__ unsigned char s_digits[ radixSortTileSize ];

   // load the scanned base offsets of this block's digit counts
   for( int d = threadIdx.x; d < radixSortBuckets; d += blockDim.x )
      s_base[ d ] = histograms[ d * static_cast< Index >( gridDim.x ) + static_cast< Index >( blockIdx.x ) ];
   // zero the counts of the elements already processed by this block
   for( int d = threadIdx.x; d < radixSortBuckets; d += blockDim.x )
      s_offset[ d ] = 0;
   __syncthreads();

   const Index n = input.getSize();
   const Index tilesPerBlock = ( numTiles + static_cast< Index >( gridDim.x ) - 1 ) / static_cast< Index >( gridDim.x );
   const Index firstTile = static_cast< Index >( blockIdx.x ) * tilesPerBlock;
   const Index lastTile = TNL::min( numTiles, firstTile + tilesPerBlock );

   for( Index tile = firstTile; tile < lastTile; tile++ ) {
      const Index tileBegin = tile * static_cast< Index >( radixSortTileSize );
      const Index tileEnd = TNL::min( n, tileBegin + radixSortTileSize );
      const Index tileSize = tileEnd - tileBegin;

      // zero the per-tile digit counts
      for( int d = threadIdx.x; d < radixSortBuckets; d += blockDim.x )
         s_counts[ d ] = 0;
      __syncthreads();

      // each thread handles the element of the tile with its index
      // (the launch uses blockSize.x == radixSortTileSize)
      const bool active = static_cast< Index >( threadIdx.x ) < tileSize;
      unsigned char myDigit = 0;
      if( active ) {
         myDigit = radixSortDigit< Value, descending >( input[ tileBegin + threadIdx.x ], pass );
         s_digits[ threadIdx.x ] = myDigit;
         atomicAdd( &s_counts[ myDigit ], 1u );
      }
      __syncthreads();

      if( active ) {
         // deterministic rank of this element among the elements
         // with the same digit in this tile
         Index rank = 0;
         for( int j = 0; j < static_cast< int >( threadIdx.x ); j++ )
            if( s_digits[ j ] == myDigit )
               rank++;
         const Index pos = s_base[ myDigit ] + s_offset[ myDigit ] + rank;
         output[ pos ] = input[ tileBegin + threadIdx.x ];
      }
      __syncthreads();

      // advance the same-digit offsets by the counts of this tile
      for( int d = threadIdx.x; d < radixSortBuckets; d += blockDim.x )
         s_offset[ d ] += static_cast< Index >( s_counts[ d ] );
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

   const Index numTiles = ( n + static_cast< Index >( radixSortTileSize ) - 1 ) / static_cast< Index >( radixSortTileSize );
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

   for( int pass = 0; pass < passes; pass++ ) {
      constexpr auto histogramKernel = radixSortHistogramKernel< Value, Index, descending >;
      Backend::launchKernelAsync( histogramKernel, launch_config, input, histograms.getView(), numTiles, pass );

      // exclusive scan of the per-block histograms turns the counts into
      // the base offsets for the scatter kernel
      Algorithms::inplaceExclusiveScan( histograms );

      constexpr auto scatterKernel = radixSortScatterKernel< Value, Index, descending >;
      Backend::launchKernelAsync( scatterKernel, launch_config, input, output, histograms.getView(), numTiles, pass );

      // the scan is host-orchestrated, so the passes cannot overlap anyway
      Backend::streamSynchronize( launch_config.stream );
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
