// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernel for the Uniform and UniformWarp thread mappings, traversing the segments given
 * explicitly by \e segmentIndexes.
 *
 * The same, fixed number of threads (\e threadsPerSegment) is assigned to every segment in the
 * launch, following the same "uniform vector width per launch" idea as
 * \ref reduceSegments_CSR_Uniform and \ref reduceSegments_CSR_Uniform_MultipleWarps.
 * Unlike those two, traversal performs no reduction - each thread just calls \e function on its
 * own elements, there is no partial result to combine across warps - so a single kernel handles
 * every width of \e threadsPerSegment (a runtime value here, not a compile-time template
 * parameter as in the reducing kernels) without needing a separate "MultipleWarps" variant for
 * segments wider than one warp.
 *
 * The choice of a fixed, uniform \e threadsPerSegment for the whole launch is inspired by the
 * same paper as \ref reduceSegments_CSR_Uniform:
 *
 * Y. Liu and B. Schmidt, "LightSpMV: Faster CSR-based sparse matrix-vector multiplication on CUDA-enabled GPUs," 2015 IEEE 26th
 * International Conference on Application-specific Systems, Architectures and Processors (ASAP), Toronto, ON, Canada, 2015, pp.
 * 82-89.
 *
 * The multi-warp shared-memory combination technique behind \ref reduceSegments_CSR_Uniform_MultipleWarps
 * does not have a counterpart here, since there is nothing to combine.
 */
template< typename OffsetsView, typename ArrayView, typename Index, typename Function >
__global__
void
forElements_CSR_Uniform_WithIndexes(
   const Index gridIdx,
   const Index threadsPerSegment,
   const OffsetsView offsets,
   const ArrayView segmentIndexes,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   const Index idx = Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
   if( idx >= segmentIndexes.getSize() )
      return;
   TNL_ASSERT_GE( idx, 0, "" );
   TNL_ASSERT_LT( idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ idx ];
   TNL_ASSERT_GE( segmentIdx, 0, "Wrong index segment index - smaller that 0." );
   TNL_ASSERT_LT( segmentIdx, offsets.getSize() - 1, "Wrong index segment index - larger that the number of indexes." );

   const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      if constexpr( callableArgumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
      localIdx += threadsPerSegment;
   }
#endif
}

/**
 * \brief Kernel for the Uniform and UniformWarp thread mappings, traversing a contiguous range
 * of segments that satisfy \e condition.
 *
 * This also implements the plain \e forElements (without a condition), which calls this kernel
 * with a trivially-true \e condition. See \ref forElements_CSR_Uniform_WithIndexes for why a
 * single kernel handles every \e threadsPerSegment width here, and for the paper that inspired
 * the uniform-width-per-launch strategy.
 */
template< typename OffsetsView, typename Index, typename Condition, typename Function >
__global__
void
forElements_CSR_Uniform(
   const Index gridIdx,
   const Index threadsPerSegment,
   const OffsetsView offsets,
   const Index begin,
   const Index end,
   Condition condition,
   Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / threadsPerSegment;
   if( segmentIdx >= end || ! condition( segmentIdx ) )
      return;

   const Index laneIdx = threadIdx.x & ( threadsPerSegment - 1 );  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += threadsPerSegment ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      if constexpr( callableArgumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
      localIdx += threadsPerSegment;
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
