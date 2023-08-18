// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/Segments/detail/BiEllpack.h>

#include "BiEllpackKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< int BlockDim,
          typename SegmentsView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__device__
void
reduceSegmentsKernelWithAllParameters( SegmentsView segments,
                                       Index gridIdx,
                                       Index begin,
                                       Index end,
                                       Fetch fetch,
                                       Reduction reduction,
                                       ResultKeeper keeper,
                                       Value identity )
{
#ifdef __CUDACC__
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   const Index segmentIdx = ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x + begin;
   if( segmentIdx >= end )
      return;

   const Index strip = segmentIdx / SegmentsView::getWarpSize();
   const Index firstGroupInStrip = strip * ( SegmentsView::getLogWarpSize() + 1 );
   const Index rowStripPerm = segments.getRowPermArrayView()[ segmentIdx ] - strip * SegmentsView::getWarpSize();
   const Index groupsCount =
      Segments::detail::BiEllpack< Index, Devices::Cuda, SegmentsView::getOrganization(), SegmentsView::getWarpSize() >::
         getActiveGroupsCountDirect( segments.getRowPermArrayView(), segmentIdx );
   Index groupHeight = SegmentsView::getWarpSize();
   bool compute = true;
   Index localIdx = 0;
   ReturnType result = identity;
   for( Index groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount && compute; groupIdx++ ) {
      Index groupOffset = segments.getGroupPointersView()[ groupIdx ];
      const Index groupSize = segments.getGroupPointersView()[ groupIdx + 1 ] - groupOffset;
      if( groupSize ) {
         const Index groupWidth = groupSize / groupHeight;
         for( Index i = 0; i < groupWidth; i++ ) {
            if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder )
               result =
                  reduction( result, fetch( segmentIdx, localIdx, groupOffset + rowStripPerm * groupWidth + i, compute ) );
            else
               result =
                  reduction( result, fetch( segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight, compute ) );
            localIdx++;
         }
      }
      groupHeight /= 2;
   }
   keeper( segmentIdx, result );
#endif
}

template< int BlockDim,
          typename SegmentsView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__device__
void
reduceSegmentsKernel( SegmentsView segments,
                      Index gridIdx,
                      Index begin,
                      Index end,
                      Fetch fetch,
                      Reduction reduction,
                      ResultKeeper keeper,
                      Value identity )
{
#ifdef __CUDACC__
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   Index segmentIdx = ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x + begin;

   const Index strip = segmentIdx >> SegmentsView::getLogWarpSize();
   const Index warpStart = strip << SegmentsView::getLogWarpSize();
   const Index inWarpIdx = segmentIdx & ( SegmentsView::getWarpSize() - 1 );

   if( warpStart >= end )
      return;

   const int warpIdx = threadIdx.x / SegmentsView::getWarpSize();
   const int warpsCount = BlockDim / SegmentsView::getWarpSize();
   constexpr int groupsInStrip = 6;  // SegmentsView::getLogWarpSize() + 1;
   // Index firstGroupIdx = strip * groupsInStrip;
   Index firstGroupInBlock = 8 * ( strip / 8 ) * groupsInStrip;
   Index groupHeight = SegmentsView::getWarpSize();

   /////
   // Allocate shared memory
   __shared__ ReturnType results[ BlockDim ];
   results[ threadIdx.x ] = identity;
   __shared__ Index sharedGroupPointers[ groupsInStrip * warpsCount + 1 ];

   /////
   // Fetch group pointers to shared memory
   // bool b1 = ( threadIdx.x <= warpsCount * groupsInStrip );
   // bool b2 = ( firstGroupIdx + threadIdx.x % groupsInStrip < segments.getGroupPointersView().getSize() );
   // printf( "tid = %d warpsCount * groupsInStrip = %d firstGroupIdx + threadIdx.x = %d
   // segments.getGroupPointersView().getSize() = %d read = %d %d\n",
   //   threadIdx.x, warpsCount * groupsInStrip,
   //   firstGroupIdx + threadIdx.x,
   //   segments.getGroupPointersView().getSize(), ( int ) b1, ( int ) b2 );
   if( threadIdx.x <= warpsCount * groupsInStrip
       && firstGroupInBlock + threadIdx.x < segments.getGroupPointersView().getSize() )
   {
      sharedGroupPointers[ threadIdx.x ] = segments.getGroupPointersView()[ firstGroupInBlock + threadIdx.x ];
      // printf( " sharedGroupPointers[ %d ] = %d \n",
      //    threadIdx.x, sharedGroupPointers[ threadIdx.x ] );
   }
   const Index sharedGroupOffset = warpIdx * groupsInStrip;
   __syncthreads();

   /////
   // Perform the reduction
   bool compute = true;
   if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
      for( Index group = 0; group < SegmentsView::getLogWarpSize() + 1; group++ ) {
         Index groupBegin = sharedGroupPointers[ sharedGroupOffset + group ];
         Index groupEnd = sharedGroupPointers[ sharedGroupOffset + group + 1 ];
         TNL_ASSERT_LT( groupBegin, segments.getStorageSize(), "" );
         // if( groupBegin >= segments.getStorageSize() )
         //    printf( "tid = %d sharedGroupOffset + group + 1 = %d strip = %d group = %d groupBegin = %d groupEnd = %d
         //    segments.getStorageSize() = %d\n",
         //       threadIdx.x, sharedGroupOffset + group + 1, strip, group, groupBegin, groupEnd, segments.getStorageSize() );
         TNL_ASSERT_LT( groupEnd, segments.getStorageSize(), "" );
         if( groupEnd - groupBegin > 0 ) {
            if( inWarpIdx < groupHeight ) {
               const Index groupWidth = ( groupEnd - groupBegin ) / groupHeight;
               Index globalIdx = groupBegin + inWarpIdx * groupWidth;
               for( Index i = 0; i < groupWidth && compute; i++ ) {
                  TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
                  results[ threadIdx.x ] = reduction( results[ threadIdx.x ], fetch( globalIdx++, compute ) );
                  // if( strip == 1 )
                  //   printf( "tid = %d i = %d groupHeight = %d groupWidth = %d globalIdx = %d fetch = %f results = %f \n",
                  //       threadIdx.x, i,
                  //       groupHeight, groupWidth,
                  //       globalIdx, fetch( globalIdx, compute ), results[ threadIdx.x ] );
               }
            }
         }
         groupHeight >>= 1;
      }
   }
   else {
      ReturnType* temp = Backend::getSharedMemory< ReturnType >();
      for( Index group = 0; group < SegmentsView::getLogWarpSize() + 1; group++ ) {
         Index groupBegin = sharedGroupPointers[ sharedGroupOffset + group ];
         Index groupEnd = sharedGroupPointers[ sharedGroupOffset + group + 1 ];
         // if( threadIdx.x < 36 && strip == 1 )
         //    printf( " tid = %d strip = %d group = %d groupBegin = %d groupEnd = %d \n", threadIdx.x, strip, group,
         //    groupBegin, groupEnd );
         if( groupEnd - groupBegin > 0 ) {
            temp[ threadIdx.x ] = identity;
            Index globalIdx = groupBegin + inWarpIdx;
            while( globalIdx < groupEnd ) {
               temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], fetch( globalIdx, compute ) );
               // if( strip == 1 )
               //    printf( "tid %d fetch %f temp %f \n", threadIdx.x, fetch( globalIdx, compute ), temp[ threadIdx.x ] );
               globalIdx += SegmentsView::getWarpSize();
            }
            // TODO: reduction via templates
            /*Index bisection2 = SegmentsView::getWarpSize();
            for( Index i = 0; i < group; i++ )
            {
               bisection2 >>= 1;
               if( inWarpIdx < bisection2 )
                  temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + bisection2 ] );
            }*/

            __syncwarp();
            if( group > 0 && inWarpIdx < 16 )
               temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 16 ] );
            __syncwarp();
            if( group > 1 && inWarpIdx < 8 )
               temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 8 ] );
            __syncwarp();
            if( group > 2 && inWarpIdx < 4 )
               temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 4 ] );
            __syncwarp();
            if( group > 3 && inWarpIdx < 2 )
               temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 2 ] );
            __syncwarp();
            if( group > 4 && inWarpIdx < 1 )
               temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 1 ] );
            __syncwarp();

            if( inWarpIdx < groupHeight )
               results[ threadIdx.x ] = reduction( results[ threadIdx.x ], temp[ threadIdx.x ] );
         }
         groupHeight >>= 1;
      }
   }
   __syncthreads();
   if( warpStart + inWarpIdx >= end )
      return;

   /////
   // Store the results
   // if( strip == 1 )
   //   printf( "Adding %f at %d \n", results[ segments.getRowPermArrayView()[ warpStart + inWarpIdx ] & ( blockDim.x - 1 ) ],
   //   warpStart + inWarpIdx );
   keeper( warpStart + inWarpIdx, results[ segments.getRowPermArrayView()[ warpStart + inWarpIdx ] & ( blockDim.x - 1 ) ] );
#endif
}

template< typename SegmentsView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value,
          int BlockDim >
__global__
void
BiEllpackreduceSegmentsKernel( SegmentsView segments,
                               Index gridIdx,
                               Index begin,
                               Index end,
                               Fetch fetch,
                               Reduction reduction,
                               ResultKeeper keeper,
                               Value identity )
{
   if constexpr( detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() )
      reduceSegmentsKernelWithAllParameters< BlockDim >( segments, gridIdx, begin, end, fetch, reduction, keeper, identity );
   else
      reduceSegmentsKernel< BlockDim >( segments, gridIdx, begin, end, fetch, reduction, keeper, identity );
}

template< typename Index, typename Device >
template< typename Segments >
void
BiEllpackKernel< Index, Device >::init( const Segments& segments )
{}

template< typename Index, typename Device >
void
BiEllpackKernel< Index, Device >::reset()
{}

template< typename Index, typename Device >
__cuda_callable__
auto
BiEllpackKernel< Index, Device >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device >
__cuda_callable__
auto
BiEllpackKernel< Index, Device >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device >
std::string
BiEllpackKernel< Index, Device >::getKernelType()
{
   return "BiEllpack";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
BiEllpackKernel< Index, Device >::reduceSegments( const SegmentsView& segments,
                                                  Index begin,
                                                  Index end,
                                                  Fetch& fetch,
                                                  const Reduction& reduction,
                                                  ResultKeeper& keeper,
                                                  const Value& identity )
{
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if( segments.getStorageSize() == 0 )
      return;
   if constexpr( std::is_same< DeviceType, Devices::Host >::value ) {
      for( IndexType segmentIdx = 0; segmentIdx < segments.getSize(); segmentIdx++ ) {
         const IndexType stripIdx = segmentIdx / SegmentsView::getWarpSize();
         const IndexType groupIdx = stripIdx * ( SegmentsView::getLogWarpSize() + 1 );
         const IndexType inStripIdx = segments.getRowPermArrayView()[ segmentIdx ] - stripIdx * SegmentsView::getWarpSize();
         const IndexType groupsCount =
            Segments::detail::BiEllpack< IndexType, DeviceType, SegmentsView::getOrganization(), SegmentsView::getWarpSize() >::
               getActiveGroupsCount( segments.getRowPermArrayView(), segmentIdx );
         IndexType globalIdx = segments.getGroupPointersView()[ groupIdx ];
         IndexType groupHeight = SegmentsView::getWarpSize();
         IndexType localIdx = 0;
         ReturnType aux = identity;
         bool compute = true;
         // std::cerr << "segmentIdx = " << segmentIdx
         //           << " stripIdx = " << stripIdx
         //           << " inStripIdx = " << inStripIdx
         //           << " groupIdx = " << groupIdx
         //          << " groupsCount = " << groupsCount
         //           << std::endl;
         for( IndexType group = 0; group < groupsCount && compute; group++ ) {
            const IndexType groupSize = Segments::detail::
               BiEllpack< IndexType, DeviceType, SegmentsView::getOrganization(), SegmentsView::getWarpSize() >::getGroupSize(
                  segments.getGroupPointersView(), stripIdx, group );
            IndexType groupWidth = groupSize / groupHeight;
            const IndexType globalIdxBack = globalIdx;
            // std::cerr << "  groupSize = " << groupSize
            //           << " groupWidth = " << groupWidth
            //           << std::endl;
            if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder )
               globalIdx += inStripIdx * groupWidth;
            else
               globalIdx += inStripIdx;
            for( IndexType j = 0; j < groupWidth && compute; j++ ) {
               // std::cerr << "    segmentIdx = " << segmentIdx << " groupIdx = " << groupIdx
               //          << " groupWidth = " << groupWidth << " groupHeight = " << groupHeight
               //           << " localIdx = " << localIdx << " globalIdx = " << globalIdx
               //           << " fetch = " << detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx,
               //           localIdx++, globalIdx, compute ) << std::endl;
               aux = reduction(
                  aux,
                  detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
               if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder )
                  globalIdx++;
               else
                  globalIdx += groupHeight;
            }
            globalIdx = globalIdxBack + groupSize;
            groupHeight /= 2;
         }
         keeper( segmentIdx, aux );
      }
   }
   if constexpr( std::is_same< DeviceType, Devices::Cuda >::value ) {
      Backend::LaunchConfiguration launch_config;
      constexpr int BlockDim = 256;
      launch_config.blockSize.x = BlockDim;
      const IndexType stripsCount = roundUpDivision( end - begin, SegmentsView::getWarpSize() );
      const IndexType cudaBlocks = roundUpDivision( stripsCount * SegmentsView::getWarpSize(), launch_config.blockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Backend::getMaxGridXSize() );
      if( SegmentsView::getOrganization() == Segments::ColumnMajorOrder )
         launch_config.dynamicSharedMemorySize = launch_config.blockSize.x * sizeof( ReturnType );

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ ) {
         launch_config.gridSize.x = Backend::getMaxGridXSize();
         if( gridIdx == cudaGrids - 1 )
            launch_config.gridSize.x = cudaBlocks % Backend::getMaxGridXSize();
         using ConstSegmentsView = typename SegmentsView::ConstViewType;
         constexpr auto kernel =
            BiEllpackreduceSegmentsKernel< ConstSegmentsView, IndexType, Fetch, Reduction, ResultKeeper, Value, BlockDim >;
         Backend::launchKernelAsync(
            kernel, launch_config, segments.getConstView(), gridIdx, begin, end, fetch, reduction, keeper, identity );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
BiEllpackKernel< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                     Fetch& fetch,
                                                     const Reduction& reduction,
                                                     ResultKeeper& keeper,
                                                     const Value& identity )
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, identity );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
