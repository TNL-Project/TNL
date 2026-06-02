// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/detail/BiEllpack.h>
#include <TNL/Algorithms/Segments/detail/FetchLambdaAdapter.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

template<
   int BlockDim,
   typename SegmentsView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__device__
void
reduceSegmentsKernelWithAllParameters(
   SegmentsView segments,
   Index gridIdx,
   Index begin,
   Index end,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   const Index segmentIdx = ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x + begin;
   const bool active = ( segmentIdx < end );

   ReturnType result = identity;
   if( active ) {
      const Index strip = segmentIdx / SegmentsView::getWarpSize();
      const Index firstGroupInStrip = strip * ( SegmentsView::getLogWarpSize() + 1 );
      const Index segmentStripPerm = segments.getSegmentsPermutationView()[ segmentIdx ] - strip * SegmentsView::getWarpSize();
      const Index groupsCount =
         Segments::detail::BiEllpack< Index, Devices::Cuda, SegmentsView::getOrganization(), SegmentsView::getWarpSize() >::
            getActiveGroupsCountDirect( segments.getSegmentsPermutationView(), segmentIdx );
      Index groupHeight = SegmentsView::getWarpSize();
      Index localIdx = 0;
      for( Index groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
         Index groupOffset = segments.getGroupPointersView()[ groupIdx ];
         const Index groupSize = segments.getGroupPointersView()[ groupIdx + 1 ] - groupOffset;
         if( groupSize ) {
            const Index groupWidth = groupSize / groupHeight;
            for( Index i = 0; i < groupWidth; i++ ) {
               if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder )
                  result = reduction( result, fetch( segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i ) );
               else
                  result = reduction( result, fetch( segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight ) );
               localIdx++;
            }
         }
         groupHeight /= 2;
      }
   }
   if( active )
      storer( segmentIdx, result );
#endif
}

template<
   int BlockDim,
   typename SegmentsView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__device__
void
reduceSegmentsKernel(
   SegmentsView segments,
   Index gridIdx,
   Index begin,
   Index end,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   Index segmentIdx = ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x + begin;

   const Index strip = segmentIdx >> SegmentsView::getLogWarpSize();
   const Index warpStart = strip << SegmentsView::getLogWarpSize();
   const Index inWarpIdx = segmentIdx & ( SegmentsView::getWarpSize() - 1 );

   const bool active = ( warpStart < end );

   const int warpIdx = threadIdx.x / SegmentsView::getWarpSize();
   const int warpsCount = BlockDim / SegmentsView::getWarpSize();
   constexpr int groupsInStrip = SegmentsView::getLogWarpSize() + 1;
   Index firstGroupInBlock = 8 * ( strip / 8 ) * groupsInStrip;
   Index groupHeight = SegmentsView::getWarpSize();

   // Allocate shared memory
   __shared__ ReturnType results[ BlockDim ];
   results[ threadIdx.x ] = identity;
   __shared__ Index sharedGroupPointers[ groupsInStrip * warpsCount + 1 ];

   // Fetch group pointers to shared memory
   if( threadIdx.x <= warpsCount * groupsInStrip
       && firstGroupInBlock + threadIdx.x < segments.getGroupPointersView().getSize() )
   {
      sharedGroupPointers[ threadIdx.x ] = segments.getGroupPointersView()[ firstGroupInBlock + threadIdx.x ];
   }
   const Index sharedGroupOffset = warpIdx * groupsInStrip;
   __syncthreads();

   // Perform the reduction
   if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
      if( active ) {
         for( Index group = 0; group < SegmentsView::getLogWarpSize() + 1; group++ ) {
            Index groupBegin = sharedGroupPointers[ sharedGroupOffset + group ];
            Index groupEnd = sharedGroupPointers[ sharedGroupOffset + group + 1 ];
            TNL_ASSERT_LE( groupBegin, segments.getStorageSize(), "" );
            TNL_ASSERT_LE( groupEnd, segments.getStorageSize(), "" );
            if( groupEnd - groupBegin > 0 ) {
               if( inWarpIdx < groupHeight ) {
                  const Index groupWidth = ( groupEnd - groupBegin ) / groupHeight;
                  Index globalIdx = groupBegin + inWarpIdx * groupWidth;
                  for( Index i = 0; i < groupWidth; i++ ) {
                     TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
                     results[ threadIdx.x ] = reduction( results[ threadIdx.x ], fetch( globalIdx++ ) );
                  }
               }
            }
            groupHeight >>= 1;
         }
      }
   }
   else {
      if( active ) {
         ReturnType* temp = Backend::getSharedMemory< ReturnType >();
         auto warp = cg::tiled_partition< SegmentsView::getWarpSize() >( cg::this_thread_block() );
         for( Index group = 0; group < SegmentsView::getLogWarpSize() + 1; group++ ) {
            Index groupBegin = sharedGroupPointers[ sharedGroupOffset + group ];
            Index groupEnd = sharedGroupPointers[ sharedGroupOffset + group + 1 ];
            if( groupEnd - groupBegin > 0 ) {
               temp[ threadIdx.x ] = identity;
               Index globalIdx = groupBegin + inWarpIdx;
               while( globalIdx < groupEnd ) {
                  temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], fetch( globalIdx ) );
                  globalIdx += SegmentsView::getWarpSize();
               }

               warp.sync();
               #pragma unroll
               for( int step = 0; step < SegmentsView::getLogWarpSize(); step++ ) {
                  const int threshold = SegmentsView::getWarpSize() >> ( step + 1 );
                  if( group > step && inWarpIdx < threshold )
                     temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + threshold ] );
                  warp.sync();
               }

               if( inWarpIdx < groupHeight )
                  results[ threadIdx.x ] = reduction( results[ threadIdx.x ], temp[ threadIdx.x ] );
            }
            groupHeight >>= 1;
         }
      }
   }
   __syncthreads();

   // Store the results - only active threads with valid segment index
   const bool storeActive = active && ( warpStart + inWarpIdx < end );
   if( storeActive ) {
      storer(
         warpStart + inWarpIdx,
         results[ segments.getSegmentsPermutationView()[ warpStart + inWarpIdx ] & ( blockDim.x - 1 ) ] );
   }
#endif
}

template<
   typename SegmentsView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value,
   int BlockDim >
__global__
void
BiEllpackReduceSegmentsKernel(
   SegmentsView segments,
   Index gridIdx,
   Index begin,
   Index end,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
   if constexpr( callableArgumentCount< Fetch >() == 3 )
      reduceSegmentsKernelWithAllParameters< BlockDim >( segments, gridIdx, begin, end, fetch, reduction, storer, identity );
   else
      reduceSegmentsKernel< BlockDim >( segments, gridIdx, begin, end, fetch, reduction, storer, identity );
}

template<
   typename SegmentsView,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value,
   int BlockDim >
__global__
void
BiEllpackReduceSegmentsKernelWithIndexes(
   SegmentsView segments,
   ArrayView segmentIndexes,
   Index gridIdx,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   const Index segmentIdx_idx = ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const bool active = ( segmentIdx_idx < segmentIndexes.getSize() );

   ReturnType result = identity;
   if( active ) {
      TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index strip = segmentIdx / SegmentsView::getWarpSize();
      const Index firstGroupInStrip = strip * ( SegmentsView::getLogWarpSize() + 1 );
      const Index segmentStripPerm = segments.getSegmentsPermutationView()[ segmentIdx ] - strip * SegmentsView::getWarpSize();
      const Index groupsCount =
         Segments::detail::BiEllpack< Index, Devices::Cuda, SegmentsView::getOrganization(), SegmentsView::getWarpSize() >::
            getActiveGroupsCountDirect( segments.getSegmentsPermutationView(), segmentIdx );
      Index groupHeight = SegmentsView::getWarpSize();
      Index localIdx = 0;
      for( Index groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
         Index groupOffset = segments.getGroupPointersView()[ groupIdx ];
         const Index groupSize = segments.getGroupPointersView()[ groupIdx + 1 ] - groupOffset;
         if( groupSize ) {
            const Index groupWidth = groupSize / groupHeight;
            for( Index i = 0; i < groupWidth; i++ ) {
               if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder )
                  result = reduction(
                     result,
                     FetchLambdaAdapter< Index, Fetch >::call(
                        fetch, segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i ) );
               else
                  result = reduction(
                     result,
                     FetchLambdaAdapter< Index, Fetch >::call(
                        fetch, segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight ) );
               localIdx++;
            }
         }
         groupHeight /= 2;
      }
      storer( segmentIdx_idx, segmentIdx, result );
   }
#endif
}

template<
   int BlockDim,
   typename SegmentsView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__device__
void
reduceSegmentsKernelWithAllParametersWithArgument(
   SegmentsView segments,
   Index gridIdx,
   Index begin,
   Index end,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   const Index segmentIdx = ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x + begin;
   const bool active = ( segmentIdx < end );

   Index argument = 0;
   ReturnType result = identity;
   if( active ) {
      const Index strip = segmentIdx / SegmentsView::getWarpSize();
      const Index firstGroupInStrip = strip * ( SegmentsView::getLogWarpSize() + 1 );
      const Index segmentStripPerm = segments.getSegmentsPermutationView()[ segmentIdx ] - strip * SegmentsView::getWarpSize();
      const Index groupsCount =
         Segments::detail::BiEllpack< Index, Devices::Cuda, SegmentsView::getOrganization(), SegmentsView::getWarpSize() >::
            getActiveGroupsCountDirect( segments.getSegmentsPermutationView(), segmentIdx );
      Index groupHeight = SegmentsView::getWarpSize();
      Index localIdx = 0;
      for( Index groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
         Index groupOffset = segments.getGroupPointersView()[ groupIdx ];
         const Index groupSize = segments.getGroupPointersView()[ groupIdx + 1 ] - groupOffset;
         if( groupSize ) {
            const Index groupWidth = groupSize / groupHeight;
            for( Index i = 0; i < groupWidth; i++ ) {
               if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder )
                  reduction(
                     result,
                     detail::FetchLambdaAdapter< Index, Fetch >::call(
                        fetch, segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i ),
                     argument,
                     localIdx );
               else
                  reduction(
                     result,
                     detail::FetchLambdaAdapter< Index, Fetch >::call(
                        fetch, segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight ),
                     argument,
                     localIdx );
               localIdx++;
            }
         }
         groupHeight /= 2;
      }
   }
   if( active ) {
      bool emptySegment = false;
      storer( segmentIdx, argument, result, emptySegment );
   }
#endif
}

template<
   typename SegmentsView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value,
   int BlockDim >
__global__
void
BiEllpackReduceSegmentsKernelWithArgument(
   SegmentsView segments,
   Index gridIdx,
   Index begin,
   Index end,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
   //Currently we do not have specialized kernel for short fetch with argument
   reduceSegmentsKernelWithAllParametersWithArgument< BlockDim >(
      segments, gridIdx, begin, end, fetch, reduction, storer, identity );
}

template<
   typename SegmentsView,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value,
   int BlockDim >
__global__
void
BiEllpackReduceSegmentsKernelWithIndexesAndArgument(
   SegmentsView segments,
   ArrayView segmentIndexes,
   Index gridIdx,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   const Index segmentIdx_idx = ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const bool active = ( segmentIdx_idx < segmentIndexes.getSize() );

   ReturnType result = identity;
   Index argument = 0;
   if( active ) {
      TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index strip = segmentIdx / SegmentsView::getWarpSize();
      const Index firstGroupInStrip = strip * ( SegmentsView::getLogWarpSize() + 1 );
      const Index segmentStripPerm = segments.getSegmentsPermutationView()[ segmentIdx ] - strip * SegmentsView::getWarpSize();
      const Index groupsCount =
         Segments::detail::BiEllpack< Index, Devices::Cuda, SegmentsView::getOrganization(), SegmentsView::getWarpSize() >::
            getActiveGroupsCountDirect( segments.getSegmentsPermutationView(), segmentIdx );
      Index groupHeight = SegmentsView::getWarpSize();
      Index localIdx = 0;
      for( Index groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
         Index groupOffset = segments.getGroupPointersView()[ groupIdx ];
         const Index groupSize = segments.getGroupPointersView()[ groupIdx + 1 ] - groupOffset;
         if( groupSize ) {
            const Index groupWidth = groupSize / groupHeight;
            for( Index i = 0; i < groupWidth; i++ ) {
               if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder )
                  reduction(
                     result,
                     FetchLambdaAdapter< Index, Fetch >::call(
                        fetch, segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i ),
                     argument,
                     localIdx );
               else
                  reduction(
                     result,
                     FetchLambdaAdapter< Index, Fetch >::call(
                        fetch, segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight ),
                     argument,
                     localIdx );
               localIdx++;
            }
         }
         groupHeight /= 2;
      }
      bool emptySegment = false;
      storer( segmentIdx_idx, segmentIdx, argument, result, emptySegment );
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
