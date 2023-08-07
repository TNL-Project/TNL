// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/KernelLaunch.h>
#include <TNL/Cuda/LaunchHelpers.h>

#include "CSRScalarKernel.h"
#include "CSRHybridKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< int ThreadsPerSegment,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
reduceSegmentsCSRHybridVectorKernel( int gridIdx,
                                     const Offsets offsets,
                                     Index begin,
                                     Index end,
                                     Fetch fetch,
                                     const Reduction reduce,
                                     ResultKeeper keep,
                                     const Value identity )
{
#ifdef __CUDACC__
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx = TNL::Cuda::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx >= end )
      return;

   const int laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   ReturnType aux = identity;
   bool compute = true;
   for( Index globalIdx = offsets[ segmentIdx ] + localIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
      aux = reduce( aux, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx, compute ) );
      localIdx += TNL::Cuda::getWarpSize();
   }

   /****
    * Reduction in each segment.
    */
   if( ThreadsPerSegment == 32 )
      aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 16 ) );
   if( ThreadsPerSegment >= 16 )
      aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 8 ) );
   if( ThreadsPerSegment >= 8 )
      aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 4 ) );
   if( ThreadsPerSegment >= 4 )
      aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 2 ) );
   if( ThreadsPerSegment >= 2 )
      aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 1 ) );

   if( laneIdx == 0 )
      keep( segmentIdx, aux );
#endif
}

template< int BlockSize,
          int ThreadsPerSegment,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
reduceSegmentsCSRHybridMultivectorKernel( int gridIdx,
                                          const Offsets offsets,
                                          Index begin,
                                          Index end,
                                          Fetch fetch,
                                          const Reduction reduce,
                                          ResultKeeper keep,
                                          const Value identity )
{
#ifdef __CUDACC__
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx = TNL::Cuda::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx >= end )
      return;

   __shared__ ReturnType shared[ BlockSize / 32 ];
   if( threadIdx.x < BlockSize / TNL::Cuda::getWarpSize() )
      shared[ threadIdx.x ] = identity;

   const int laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );               // & is cheaper than %
   const int inWarpLaneIdx = threadIdx.x & ( TNL::Cuda::getWarpSize() - 1 );  // & is cheaper than %
   const Index beginIdx = offsets[ segmentIdx ];
   const Index endIdx = offsets[ segmentIdx + 1 ];

   ReturnType result = identity;
   bool compute = true;
   Index localIdx = laneIdx;
   for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx && compute; globalIdx += ThreadsPerSegment ) {
      result =
         reduce( result, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx, compute ) );
      localIdx += ThreadsPerSegment;
   }
   result += __shfl_down_sync( 0xFFFFFFFF, result, 16 );
   result += __shfl_down_sync( 0xFFFFFFFF, result, 8 );
   result += __shfl_down_sync( 0xFFFFFFFF, result, 4 );
   result += __shfl_down_sync( 0xFFFFFFFF, result, 2 );
   result += __shfl_down_sync( 0xFFFFFFFF, result, 1 );

   const Index warpIdx = threadIdx.x / TNL::Cuda::getWarpSize();
   if( inWarpLaneIdx == 0 )
      shared[ warpIdx ] = result;

   __syncthreads();
   // Reduction in shared
   if( warpIdx == 0 && inWarpLaneIdx < 16 ) {
      // constexpr int totalWarps = BlockSize / WarpSize;
      constexpr int warpsPerSegment = ThreadsPerSegment / TNL::Cuda::getWarpSize();
      if( warpsPerSegment >= 32 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 16 ] );
         __syncwarp();
      }
      if( warpsPerSegment >= 16 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 8 ] );
         __syncwarp();
      }
      if( warpsPerSegment >= 8 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 4 ] );
         __syncwarp();
      }
      if( warpsPerSegment >= 4 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 2 ] );
         __syncwarp();
      }
      if( warpsPerSegment >= 2 ) {
         shared[ inWarpLaneIdx ] = reduce( shared[ inWarpLaneIdx ], shared[ inWarpLaneIdx + 1 ] );
         __syncwarp();
      }
      constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
      if( inWarpLaneIdx < segmentsCount && segmentIdx + inWarpLaneIdx < end ) {
         // printf( "Long: segmentIdx %d -> %d \n", segmentIdx, aux );
         keep( segmentIdx + inWarpLaneIdx, shared[ inWarpLaneIdx * ThreadsPerSegment / 32 ] );
      }
   }
#endif
}

template< typename Index, typename Device, int ThreadsInBlock >
template< typename Segments >
void
CSRHybridKernel< Index, Device, ThreadsInBlock >::init( const Segments& segments )
{
   const auto& offsets = segments.getOffsets();
   TNL_ASSERT_GT( offsets.getSize(), 0, "offsets size must be strictly positive" );
   const Index segmentsCount = offsets.getSize() - 1;
   if( segmentsCount <= 0 )
      return;
   const Index elementsInSegment = std::ceil( (double) offsets.getElement( segmentsCount ) / (double) segmentsCount );
   this->threadsPerSegment =
      TNL::min( std::pow( 2, std::ceil( std::log2( elementsInSegment ) ) ), ThreadsInBlock );  // TNL::Cuda::getWarpSize() );
   TNL_ASSERT_GE( threadsPerSegment, 0, "" );
   TNL_ASSERT_LE( threadsPerSegment, ThreadsInBlock, "" );
}

template< typename Index, typename Device, int ThreadsInBlock >
void
CSRHybridKernel< Index, Device, ThreadsInBlock >::reset()
{
   this->threadsPerSegment = 0;
}

template< typename Index, typename Device, int ThreadsInBlock >
__cuda_callable__
auto
CSRHybridKernel< Index, Device, ThreadsInBlock >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device, int ThreadsInBlock >
std::string
CSRHybridKernel< Index, Device, ThreadsInBlock >::getKernelType()
{
   return "Hybrid " + std::to_string( ThreadsInBlock );
}

template< typename Index, typename Device, int ThreadsInBlock >
__cuda_callable__
auto
CSRHybridKernel< Index, Device, ThreadsInBlock >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device, int ThreadsInBlock >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRHybridKernel< Index, Device, ThreadsInBlock >::reduceSegments( const SegmentsView& segments,
                                                                  Index begin,
                                                                  Index end,
                                                                  Fetch& fetch,
                                                                  const Reduction& reduction,
                                                                  ResultKeeper& keeper,
                                                                  const Value& identity ) const
{
   constexpr bool DispatchScalarCSR = std::is_same< Device, Devices::Host >::value;
   if constexpr( DispatchScalarCSR ) {
      TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel< Index, Device >::reduceSegments(
         segments, begin, end, fetch, reduction, keeper, identity );
   }
   else {
      using OffsetsView = typename SegmentsView::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();

      if( end <= begin )
         return;

      Cuda::LaunchConfiguration launch_config;
      launch_config.blockSize.x = ThreadsInBlock;
      const size_t threadsCount = this->threadsPerSegment * ( end - begin );
      dim3 blocksCount;
      dim3 gridsCount;
      TNL::Cuda::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );

      for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
         TNL::Cuda::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
         switch( this->threadsPerSegment ) {
            case 0:  // this means zero/empty matrix
               break;
            case 1:
               {
                  constexpr auto kernel =
                     reduceSegmentsCSRHybridVectorKernel< 1, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Value >;
                  Cuda::launchKernelAsync(
                     kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, identity );
                  break;
               }
            case 2:
               {
                  constexpr auto kernel =
                     reduceSegmentsCSRHybridVectorKernel< 2, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Value >;
                  Cuda::launchKernelAsync(
                     kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, identity );
                  break;
               }
            case 4:
               {
                  constexpr auto kernel =
                     reduceSegmentsCSRHybridVectorKernel< 4, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Value >;
                  Cuda::launchKernelAsync(
                     kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, identity );
                  break;
               }
            case 8:
               {
                  constexpr auto kernel =
                     reduceSegmentsCSRHybridVectorKernel< 8, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Value >;
                  Cuda::launchKernelAsync(
                     kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, identity );
                  break;
               }
            case 16:
               {
                  constexpr auto kernel =
                     reduceSegmentsCSRHybridVectorKernel< 16, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Value >;
                  Cuda::launchKernelAsync(
                     kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, identity );
                  break;
               }
            case 32:
               {
                  constexpr auto kernel =
                     reduceSegmentsCSRHybridVectorKernel< 32, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Value >;
                  Cuda::launchKernelAsync(
                     kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, identity );
                  break;
               }
            case 64:
               {
                  constexpr auto kernel = reduceSegmentsCSRHybridMultivectorKernel< ThreadsInBlock,
                                                                                    64,
                                                                                    OffsetsView,
                                                                                    Index,
                                                                                    Fetch,
                                                                                    Reduction,
                                                                                    ResultKeeper,
                                                                                    Value >;
                  Cuda::launchKernelAsync(
                     kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, identity );
                  break;
               }
            case 128:
               {
                  constexpr auto kernel = reduceSegmentsCSRHybridMultivectorKernel< ThreadsInBlock,
                                                                                    128,
                                                                                    OffsetsView,
                                                                                    Index,
                                                                                    Fetch,
                                                                                    Reduction,
                                                                                    ResultKeeper,
                                                                                    Value >;
                  Cuda::launchKernelAsync(
                     kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, identity );
                  break;
               }
            case 256:
               {
                  constexpr auto kernel = reduceSegmentsCSRHybridMultivectorKernel< ThreadsInBlock,
                                                                                    256,
                                                                                    OffsetsView,
                                                                                    Index,
                                                                                    Fetch,
                                                                                    Reduction,
                                                                                    ResultKeeper,
                                                                                    Value >;
                  Cuda::launchKernelAsync(
                     kernel, launch_config, gridIdx, offsets, begin, end, fetch, reduction, keeper, identity );
                  break;
               }
            default:
               throw std::runtime_error( std::string( "Wrong value of threadsPerSegment: " )
                                         + std::to_string( this->threadsPerSegment ) );
         }
      }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< typename Index, typename Device, int ThreadsInBlock >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRHybridKernel< Index, Device, ThreadsInBlock >::reduceAllSegments( const SegmentsView& segments,
                                                                     Fetch& fetch,
                                                                     const Reduction& reduction,
                                                                     ResultKeeper& keeper,
                                                                     const Value& identity ) const
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, identity );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
