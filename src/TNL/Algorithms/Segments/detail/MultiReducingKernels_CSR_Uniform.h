// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/detail/CudaReductionKernel.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>
#include "MultiReducingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Multireduction kernel for the Uniform thread mapping which processes all tiles of
 * the reductions at once.
 *
 * The reductions are split into tiles of `TileSize` reductions, the tile `firstTile + blockIdx.y`
 * is processed by the blocks with the given `blockIdx.y`. `ThreadsPerSegment` threads (at most
 * the warp size) process every segment as in \ref reduceSegments_CSR_Uniform, each thread keeps
 * the partial results of all reductions of the tile and they are combined by warp shuffles.
 * Compared to processing the tiles one after another, all tiles run in parallel, which matters
 * for a few segments and many reductions.
 */
template<
   int ThreadsPerSegment,
   int TileSize,
   typename Segments,
   typename Index,
   typename Fetch,
   typename Reduce,
   typename Store,
   typename Value >
__global__
void
multiReduceSegments_CSR_Uniform(
   const Index gridID,
   const Segments segments,
   const Index begin,
   const Index end,
   const int firstTile,
   const int reductions,
   Fetch fetch,
   Reduce reduce,
   Store store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using MultiValue = MultiReductionValue< TileSize, Value >;
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridID ) / ThreadsPerSegment;
   const bool active = segmentIdx < end;
   const int tileBegin = ( firstTile + static_cast< int >( blockIdx.y ) ) * TileSize;
   const int tileReductions = TNL::min( TileSize, reductions - tileBegin );

   MultiValue result;
   for( int j = 0; j < TileSize; j++ )
      result[ j ] = identity;
   if( active ) {
      const int laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );
      const Index beginIdx = segments.getOffsets()[ segmentIdx ];
      const Index endIdx = segments.getOffsets()[ segmentIdx + 1 ];
      auto reduceElements = [ & ]( auto full )
      {
         for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
   #pragma unroll
            for( int j = 0; j < TileSize; j++ ) {
               if( decltype( full )::value || j < tileReductions ) {
                  if constexpr( callableArgumentCount< Fetch >() == 4 )
                     result[ j ] = reduce( result[ j ], fetch( segmentIdx, globalIdx - beginIdx, globalIdx, tileBegin + j ) );
                  else
                     result[ j ] = reduce( result[ j ], fetch( globalIdx, tileBegin + j ) );
               }
            }
         }
      };
      if( tileReductions == TileSize )
         reduceElements( std::true_type{} );
      else
         reduceElements( std::false_type{} );
   }

   // all threads must participate in the shuffles on HIP
   const MultiReductionOperation< TileSize, Reduce, Value > multiReduce{ reduce };
   using BlockReduce =
      Algorithms::detail::CudaBlockReduceShfl< 256, MultiReductionOperation< TileSize, Reduce, Value >, MultiValue >;
   result = BlockReduce::template warpReduce< ThreadsPerSegment >( multiReduce, result );

   if( active && ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 )
      for( int j = 0; j < tileReductions; j++ )
         store( segmentIdx, tileBegin + j, result[ j ] );
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
