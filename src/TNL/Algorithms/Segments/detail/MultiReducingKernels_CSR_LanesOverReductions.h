// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Backend/Functions.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )
// Reduces the elements first, first + step, ... smaller than last of the segment segmentIdx starting at the global
// index segmentBegin into the results of the reductions firstReduction, ..., firstReduction + ReductionsPerLane - 1
// (only those smaller than reductions). See multiReduceSegments_CSR_LanesOverReductions for the forms of the fetch.
template< int ReductionsPerLane, typename Index, typename Fetch, typename Reduce, typename Value >
__device__
void
reduceElementsLanesOverReductions(
   Index segmentIdx,
   Index segmentBegin,
   Index first,
   Index last,
   Index step,
   int firstReduction,
   int reductions,
   Fetch& fetch,
   Reduce& reduce,
   Value ( &results )[ ReductionsPerLane ] )
{
   auto reduceElement = [ & ]( Index globalIdx, int r )
   {
      if constexpr( callableArgumentCount< Fetch >() == 4 )
         results[ r ] = reduce( results[ r ], fetch( segmentIdx, globalIdx - segmentBegin, globalIdx, firstReduction + r ) );
      else
         results[ r ] = reduce( results[ r ], fetch( globalIdx, firstReduction + r ) );
   };
   if( firstReduction + ReductionsPerLane <= reductions ) {
      if constexpr( std::is_invocable_v< Fetch, Index, int, std::integral_constant< int, ReductionsPerLane > > ) {
         // the block fetch returns the values of all reductions of the lane at once
         for( Index globalIdx = first; globalIdx < last; globalIdx += step ) {
            const auto values = fetch( globalIdx, firstReduction, std::integral_constant< int, ReductionsPerLane >{} );
   #pragma unroll
            for( int r = 0; r < ReductionsPerLane; r++ )
               results[ r ] = reduce( results[ r ], values[ r ] );
         }
      }
      else {
         for( Index globalIdx = first; globalIdx < last; globalIdx += step ) {
   #pragma unroll
            for( int r = 0; r < ReductionsPerLane; r++ )
               reduceElement( globalIdx, r );
         }
      }
   }
   else {
      // the last lane with only some of its reductions
      for( Index globalIdx = first; globalIdx < last; globalIdx += step ) {
   #pragma unroll
         for( int r = 0; r < ReductionsPerLane; r++ )
            if( firstReduction + r < reductions )
               reduceElement( globalIdx, r );
      }
   }
}
#endif

/**
 * \brief Multireduction kernel which distributes the reductions among the lanes.
 *
 * A group of `Lanes` threads is assigned to every segment and all lanes of
 * the group traverse all elements of the segment. The lane `l` computes the
 * reductions `j = l * ReductionsPerLane, ..., (l + 1) * ReductionsPerLane - 1`
 * (only those smaller than `reductions`), so the lanes do not need to combine
 * their partial results.
 *
 * The lanes of the group read the same element of the segment at the same
 * time, which the hardware serves by one broadcast load. When the fetch of the
 * reduction `j` reads data indexed by `j` (e.g. the row of a row-major dense
 * matrix in the product of a sparse and a dense matrix), the reads of the
 * lanes are coalesced. This is the row-splitting SpMM with the row-major
 * coalesced access to the dense matrix by
 *
 * C. Yang, A. Buluç, J. D. Owens,
 * "Design Principles for Sparse Matrix Multiplication on the GPU,"
 * Euro-Par 2018, LNCS 11014, pp. 672-687.
 *
 * If the brief fetch can be called as
 * `fetch( globalIdx, firstReduction, std::integral_constant< int, ReductionsPerLane >{} )`,
 * the lanes with all their reductions valid use it to obtain the values of all their
 * reductions at once (e.g. by vectorized loads). The result must provide `operator[]`
 * for `r = 0, ..., ReductionsPerLane - 1`.
 *
 * The segments with more than `maxSegmentSize` elements are skipped and counted in
 * `longSegmentsCount`, they are processed by \ref multiReduceSegments_CSR_LanesOverReductions_Pieces.
 */
template<
   int Lanes,
   int ReductionsPerLane,
   typename Segments,
   typename Index,
   typename Fetch,
   typename Reduce,
   typename Store,
   typename Value >
__global__
void
multiReduceSegments_CSR_LanesOverReductions(
   const Index gridID,
   const Segments segments,
   const Index begin,
   const Index end,
   const Index maxSegmentSize,
   int* longSegmentsCount,
   const int reductions,
   Fetch fetch,
   Reduce reduce,
   Store store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridID ) / Lanes;
   const int firstReduction = ( threadIdx.x & ( Lanes - 1 ) ) * ReductionsPerLane;
   // there is no communication between the lanes, so the idle threads can quit
   if( segmentIdx >= end || firstReduction >= reductions )
      return;
   const Index beginIdx = segments.getOffsets()[ segmentIdx ];
   const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];
   if( endIdx - beginIdx > maxSegmentSize ) {
      if( firstReduction == 0 )
         atomicAdd( longSegmentsCount, 1 );
      return;
   }

   Value results[ ReductionsPerLane ];
   for( int r = 0; r < ReductionsPerLane; r++ )
      results[ r ] = identity;
   reduceElementsLanesOverReductions< ReductionsPerLane >(
      segmentIdx, beginIdx, beginIdx, endIdx, Index( 1 ), firstReduction, reductions, fetch, reduce, results );

   for( int r = 0; r < ReductionsPerLane; r++ )
      if( firstReduction + r < reductions )
         store( segmentIdx, firstReduction + r, results[ r ] );
#endif
}

/**
 * \brief Multireduction kernel for the pieces of long segments.
 *
 * Every thread block processes one piece, i.e. the elements `pieceBegins[ p ], ...,
 * pieceEnds[ p ] - 1` of the segment `pieceSegments[ p ]`. The block is split into
 * groups of `Lanes` threads which distribute the reductions among the lanes as in
 * \ref multiReduceSegments_CSR_LanesOverReductions and process every `BlockSize / Lanes`-th
 * element of the piece. The results of the groups are combined in the shared memory and the
 * results of the piece are written to `partialResults[ p * Lanes * ReductionsPerLane + j ]`.
 */
template<
   int BlockSize,
   int Lanes,
   int ReductionsPerLane,
   typename Segments,
   typename IndexView,
   typename ValueView,
   typename Index,
   typename Fetch,
   typename Reduce,
   typename Value >
__global__
void
multiReduceSegments_CSR_LanesOverReductions_Pieces(
   const Segments segments,
   const IndexView pieceSegments,
   const IndexView pieceBegins,
   const IndexView pieceEnds,
   const int reductions,
   ValueView partialResults,
   Fetch fetch,
   Reduce reduce,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   constexpr int groups = BlockSize / Lanes;
   constexpr int tileSize = Lanes * ReductionsPerLane;
   static_assert( tileSize <= BlockSize, "The results of the piece are written by the first tileSize threads." );
   const Index piece = blockIdx.x;
   const Index segmentIdx = pieceSegments[ piece ];
   const int group = threadIdx.x / Lanes;
   const int firstReduction = ( threadIdx.x & ( Lanes - 1 ) ) * ReductionsPerLane;

   Value results[ ReductionsPerLane ];
   for( int r = 0; r < ReductionsPerLane; r++ )
      results[ r ] = identity;
   if( firstReduction < reductions )
      reduceElementsLanesOverReductions< ReductionsPerLane >(
         segmentIdx,
         segments.getOffsets()[ segmentIdx ],
         pieceBegins[ piece ] + group,
         pieceEnds[ piece ],
         Index( groups ),
         firstReduction,
         reductions,
         fetch,
         reduce,
         results );

   // Complex has a non-trivial default constructor, which HIP rejects for __shared__ variables
   __shared__ Backend::Uninitialized< Value > shared[ BlockSize * ReductionsPerLane ];
   // the result j of the group g is stored at g * tileSize + j
   for( int r = 0; r < ReductionsPerLane; r++ )
      shared[ threadIdx.x * ReductionsPerLane + r ] = results[ r ];
   __syncthreads();
   if( threadIdx.x < tileSize && threadIdx.x < reductions ) {
      Value result = shared[ threadIdx.x ].get();
      for( int g = 1; g < groups; g++ )
         result = reduce( result, shared[ g * tileSize + threadIdx.x ].get() );
      partialResults[ piece * tileSize + threadIdx.x ] = result;
   }
#endif
}

/**
 * \brief Combines the partial results of the pieces of long segments computed by
 * \ref multiReduceSegments_CSR_LanesOverReductions_Pieces.
 *
 * The thread `i * tileSize + j` reduces the partial results of the reduction `j` over
 * the pieces `firstPieces[ i ], ..., firstPieces[ i + 1 ] - 1` of the long segment
 * `longSegments[ i ]` and stores the result.
 */
template< int TileSize, typename IndexView, typename ValueView, typename Index, typename Reduce, typename Store >
__global__
void
multiReduceSegments_CSR_LanesOverReductions_CombinePieces(
   const IndexView longSegments,
   const IndexView firstPieces,
   const int reductions,
   const ValueView partialResults,
   Reduce reduce,
   Store store )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index threadIdx_ = static_cast< Index >( blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index i = threadIdx_ / TileSize;
   const int j = threadIdx_ % TileSize;
   if( i >= longSegments.getSize() || j >= reductions )
      return;
   auto result = partialResults[ firstPieces[ i ] * TileSize + j ];
   for( Index piece = firstPieces[ i ] + 1; piece < firstPieces[ i + 1 ]; piece++ )
      result = reduce( result, partialResults[ piece * TileSize + j ] );
   store( longSegments[ i ], j, result );
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
