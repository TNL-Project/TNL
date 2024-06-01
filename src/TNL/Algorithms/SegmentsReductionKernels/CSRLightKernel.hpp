// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend.h>

#include "CSRLightKernel.h"
#include "CSRScalarKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< int ThreadsPerSegment,
          typename Value,
          typename Index,
          typename OffsetsView,
          typename Fetch,
          typename Reduce,
          typename Keep >
__global__
void
SpMVCSRVector( OffsetsView offsets,
               const Index begin,
               const Index end,
               Fetch fetch,
               Reduce reduce,
               Keep keep,
               const Value identity,
               const Index gridID )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index warpID =
      begin + ( ( gridID * Backend::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / ThreadsPerSegment;
   if( warpID >= end )
      return;

   ReturnType result = identity;
   const Index laneID = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %
   Index endID = offsets[ warpID + 1 ];

   // Calculate result
   bool compute = true;
   for( Index i = offsets[ warpID ] + laneID; i < endID; i += ThreadsPerSegment )
      result = reduce( result, fetch( i, compute ) );

   // Parallel reduction
   #if defined( __HIP__ )
   if( ThreadsPerSegment > 16 ) {
      result = reduce( result, __shfl_down( result, 16 ) );
      result = reduce( result, __shfl_down( result, 8 ) );
      result = reduce( result, __shfl_down( result, 4 ) );
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      result = reduce( result, __shfl_down( result, 8 ) );
      result = reduce( result, __shfl_down( result, 4 ) );
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      result = reduce( result, __shfl_down( result, 4 ) );
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      result = reduce( result, __shfl_down( result, 1 ) );
   #else
   if( ThreadsPerSegment > 16 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   #endif

   // Write the result
   if( laneID == 0 )
      keep( warpID, result );
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
reduceSegmentsCSRLightMultivectorKernel( int gridIdx,
                                         const Offsets offsets,
                                         Index begin,
                                         Index end,
                                         Fetch fetch,
                                         const Reduction reduce,
                                         ResultKeeper keep,
                                         const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment + begin;
   if( segmentIdx >= end )
      return;

   __shared__ ReturnType shared[ BlockSize / Backend::getWarpSize() ];
   if( threadIdx.x < BlockSize / Backend::getWarpSize() )
      shared[ threadIdx.x ] = identity;

   const int laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );             // & is cheaper than %
   const int inWarpLaneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
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

   #if defined( __HIP__ )
   result += __shfl_down( result, 16 );
   result += __shfl_down( result, 8 );
   result += __shfl_down( result, 4 );
   result += __shfl_down( result, 2 );
   result += __shfl_down( result, 1 );
   #else
   result += __shfl_down_sync( 0xFFFFFFFF, result, 16 );
   result += __shfl_down_sync( 0xFFFFFFFF, result, 8 );
   result += __shfl_down_sync( 0xFFFFFFFF, result, 4 );
   result += __shfl_down_sync( 0xFFFFFFFF, result, 2 );
   result += __shfl_down_sync( 0xFFFFFFFF, result, 1 );
   #endif

   const Index warpIdx = threadIdx.x / Backend::getWarpSize();
   if( inWarpLaneIdx == 0 )
      shared[ warpIdx ] = result;

   __syncthreads();
   // Reduction in shared
   if( warpIdx == 0 && inWarpLaneIdx < 16 ) {
      // constexpr int totalWarps = BlockSize / Backend::getWarpSize();
      constexpr int warpsPerSegment = ThreadsPerSegment / Backend::getWarpSize();
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
         keep( segmentIdx + inWarpLaneIdx, shared[ inWarpLaneIdx * ThreadsPerSegment / Backend::getWarpSize() ] );
      }
   }
#endif
}

template< typename Index, typename Device >
template< typename Segments >
void
CSRLightKernel< Index, Device >::init( const Segments& segments )
{
   const auto& offsets = segments.getOffsets();
   TNL_ASSERT_GT( offsets.getSize(), 0, "offsets size must be strictly positive" );
   const Index segmentsCount = offsets.getSize() - 1;
   if( segmentsCount <= 0 )
      return;

   if( this->getThreadsMapping() == CSRLightAutomaticThreads ) {
      const Index elementsInSegment =
         roundUpDivision( offsets.getElement( segmentsCount ), segmentsCount );  // non identityes per row
      if( elementsInSegment <= 2 )
         setThreadsPerSegment( 2 );
      else if( elementsInSegment <= 4 )
         setThreadsPerSegment( 4 );
      else if( elementsInSegment <= 8 )
         setThreadsPerSegment( 8 );
      else if( elementsInSegment <= 16 )
         setThreadsPerSegment( 16 );
      else                            // if (nnz <= 2 * matrix.MAX_ELEMENTS_PER_WARP)
         setThreadsPerSegment( 32 );  // CSR Vector
      // else
      //    threadsPerSegment = roundUpDivision(nnz, matrix.MAX_ELEMENTS_PER_WARP) * 32; // CSR MultiVector
   }

   if( this->getThreadsMapping() == CSRLightAutomaticThreadsLightSpMV ) {
      const Index elementsInSegment =
         roundUpDivision( offsets.getElement( segmentsCount ), segmentsCount );  // non identityes per row
      if( elementsInSegment <= 2 )
         setThreadsPerSegment( 2 );
      else if( elementsInSegment <= 4 )
         setThreadsPerSegment( 4 );
      else if( elementsInSegment <= 8 )
         setThreadsPerSegment( 8 );
      else if( elementsInSegment <= 16 )
         setThreadsPerSegment( 16 );
      else                            // if (nnz <= 2 * matrix.MAX_ELEMENTS_PER_WARP)
         setThreadsPerSegment( 32 );  // CSR Vector
      // else
      //    threadsPerSegment = roundUpDivision(nnz, matrix.MAX_ELEMENTS_PER_WARP) * 32; // CSR MultiVector
   }
}

template< typename Index, typename Device >
void
CSRLightKernel< Index, Device >::reset()
{
   this->threadsPerSegment = 0;
}

template< typename Index, typename Device >
__cuda_callable__
auto
CSRLightKernel< Index, Device >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device >
std::string
CSRLightKernel< Index, Device >::getKernelType()
{
   return "Light";
}

template< typename Index, typename Device >
__cuda_callable__
auto
CSRLightKernel< Index, Device >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename Keep, typename Value >
void
CSRLightKernel< Index, Device >::reduceSegments( const SegmentsView& segments,
                                                 Index begin,
                                                 Index end,
                                                 Fetch& fetch,
                                                 const Reduction& reduction,
                                                 Keep& keep,
                                                 const Value& identity ) const
{
   constexpr bool DispatchScalarCSR = detail::CheckFetchLambda< Index, Fetch >::hasAllParameters()
                                   || std::is_same_v< Device, Devices::Host > || ! std::is_fundamental_v< Value >;
   if constexpr( DispatchScalarCSR ) {
      TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel< Index, Device >::reduceSegments(
         segments, begin, end, fetch, reduction, keep, identity );
   }
   else {
      TNL_ASSERT_GE( this->threadsPerSegment, 0, "" );
      TNL_ASSERT_LE( this->threadsPerSegment, 128, "" );

      if( end <= begin )
         return;

      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 128;

      std::size_t neededThreads = threadsPerSegment * ( end - begin );

      using OffsetsView = typename SegmentsView::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();

      for( Index grid = 0; neededThreads != 0; ++grid ) {
         if( Backend::getMaxGridXSize() * launch_config.blockSize.x >= neededThreads ) {
            launch_config.gridSize.x = roundUpDivision( neededThreads, launch_config.blockSize.x );
            neededThreads = 0;
         }
         else {
            launch_config.gridSize.x = Backend::getMaxGridXSize();
            neededThreads -= Backend::getMaxGridXSize() * launch_config.blockSize.x;
         }

         if( threadsPerSegment == 1 ) {
            constexpr auto kernel = SpMVCSRVector< 1, Value, Index, OffsetsView, Fetch, Reduction, Keep >;
            Backend::launchKernelAsync( kernel, launch_config, offsets, begin, end, fetch, reduction, keep, identity, grid );
         }
         if( threadsPerSegment == 2 ) {
            constexpr auto kernel = SpMVCSRVector< 2, Value, Index, OffsetsView, Fetch, Reduction, Keep >;
            Backend::launchKernelAsync( kernel, launch_config, offsets, begin, end, fetch, reduction, keep, identity, grid );
         }
         if( threadsPerSegment == 4 ) {
            constexpr auto kernel = SpMVCSRVector< 4, Value, Index, OffsetsView, Fetch, Reduction, Keep >;
            Backend::launchKernelAsync( kernel, launch_config, offsets, begin, end, fetch, reduction, keep, identity, grid );
         }
         if( threadsPerSegment == 8 ) {
            constexpr auto kernel = SpMVCSRVector< 8, Value, Index, OffsetsView, Fetch, Reduction, Keep >;
            Backend::launchKernelAsync( kernel, launch_config, offsets, begin, end, fetch, reduction, keep, identity, grid );
         }
         if( threadsPerSegment == 16 ) {
            constexpr auto kernel = SpMVCSRVector< 16, Value, Index, OffsetsView, Fetch, Reduction, Keep >;
            Backend::launchKernelAsync( kernel, launch_config, offsets, begin, end, fetch, reduction, keep, identity, grid );
         }
         if( threadsPerSegment == 32 ) {
            constexpr auto kernel = SpMVCSRVector< 32, Value, Index, OffsetsView, Fetch, Reduction, Keep >;
            Backend::launchKernelAsync( kernel, launch_config, offsets, begin, end, fetch, reduction, keep, identity, grid );
         }
         if( threadsPerSegment == 64 ) {  // Execute CSR MultiVector
            constexpr auto kernel =
               reduceSegmentsCSRLightMultivectorKernel< 128, 64, OffsetsView, Index, Fetch, Reduction, Keep, Value >;
            Backend::launchKernelAsync( kernel, launch_config, grid, offsets, begin, end, fetch, reduction, keep, identity );
         }
         if( threadsPerSegment >= 128 ) {  // Execute CSR MultiVector
            constexpr auto kernel =
               reduceSegmentsCSRLightMultivectorKernel< 128, 128, OffsetsView, Index, Fetch, Reduction, Keep, Value >;
            Backend::launchKernelAsync( kernel, launch_config, grid, offsets, begin, end, fetch, reduction, keep, identity );
         }
      }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRLightKernel< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                    Fetch& fetch,
                                                    const Reduction& reduction,
                                                    ResultKeeper& keeper,
                                                    const Value& identity ) const
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, identity );
}

template< typename Index, typename Device >
void
CSRLightKernel< Index, Device >::setThreadsMapping( LightCSRSThreadsMapping mapping )
{
   this->mapping = mapping;
}

template< typename Index, typename Device >
LightCSRSThreadsMapping
CSRLightKernel< Index, Device >::getThreadsMapping() const
{
   return this->mapping;
}

template< typename Index, typename Device >
void
CSRLightKernel< Index, Device >::setThreadsPerSegment( int threadsPerSegment )
{
   if( threadsPerSegment != 1 && threadsPerSegment != 2 && threadsPerSegment != 4 && threadsPerSegment != 8
       && threadsPerSegment != 16 && threadsPerSegment != 32 && threadsPerSegment != 64 && threadsPerSegment != 128 )
      throw std::invalid_argument( "Number of threads per segment must be power of 2 - 1, 2, ... 128." );
   this->threadsPerSegment = threadsPerSegment;
}

template< typename Index, typename Device >
int
CSRLightKernel< Index, Device >::getThreadsPerSegment() const
{
   return this->threadsPerSegment;
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
