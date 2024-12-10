// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Backend.h>
#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/find.h>
#include <TNL/Algorithms/detail/CudaScanKernel.h>

#include "CSRBase.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index >
__cuda_callable__
void
CSRBase< Device, Index >::bind( OffsetsView offsets )
{
   this->offsets.bind( std::move( offsets ) );
}

template< typename Device, typename Index >
__cuda_callable__
CSRBase< Device, Index >::CSRBase( const OffsetsView& offsets )
: offsets( offsets )
{}

template< typename Device, typename Index >
__cuda_callable__
CSRBase< Device, Index >::CSRBase( OffsetsView&& offsets )
: offsets( std::move( offsets ) )
{}

template< typename Device, typename Index >
std::string
CSRBase< Device, Index >::getSerializationType()
{
   return "CSR< " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device, typename Index >
std::string
CSRBase< Device, Index >::getSegmentsType()
{
   return "CSR";
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getSegmentsCount() const -> IndexType
{
   return this->offsets.getSize() - 1;
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getSegmentSize( IndexType segmentIdx ) const -> IndexType
{
   if( ! std::is_same_v< DeviceType, Devices::Host > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
#else
      return offsets.getElement( segmentIdx + 1 ) - offsets.getElement( segmentIdx );
#endif
   }
   return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getSize() const -> IndexType
{
   return this->getStorageSize();
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getStorageSize() const -> IndexType
{
   if( ! std::is_same_v< DeviceType, Devices::Host > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return offsets[ getSegmentsCount() ];
#else
      return offsets.getElement( getSegmentsCount() );
#endif
   }
   return offsets[ getSegmentsCount() ];
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
   if( ! std::is_same_v< DeviceType, Devices::Host > ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return offsets[ segmentIdx ] + localIdx;
#else
      return offsets.getElement( segmentIdx ) + localIdx;
#endif
   }
   return offsets[ segmentIdx ] + localIdx;
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getSegmentView( IndexType segmentIdx ) const -> SegmentViewType
{
   return SegmentViewType( segmentIdx, offsets[ segmentIdx ], offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ] );
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getOffsets() -> OffsetsView
{
   return this->offsets;
}

template< typename Device, typename Index >
__cuda_callable__
auto
CSRBase< Device, Index >::getOffsets() const -> ConstOffsetsView
{
   return this->offsets.getConstView();
}

template< typename OffsetsView, typename Index, typename Function, int BlockSize = 256 >
__global__
void
forElementsBlockMergeKernel( Index gridIdx, OffsetsView offsets, Index begin, Index end, Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   __shared__ Index shared_offsets[ BlockSize + 1 ];
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx );
   if( segmentIdx <= end )
      shared_offsets[ threadIdx.x ] = offsets[ segmentIdx ];
   if( threadIdx.x == 0 && end - begin >= BlockSize )
      shared_offsets[ BlockSize ] = offsets[ end ];
   __syncthreads();

   const Index first_segment_in_block = segmentIdx - threadIdx.x;
   const Index last_segment_in_block = min( end, first_segment_in_block + BlockSize );
   const Index segments_in_block = last_segment_in_block - first_segment_in_block;
   const Index first_idx = shared_offsets[ 0 ];
   const Index last_idx = offsets[ last_segment_in_block ];

   Index idx = threadIdx.x;
   while( idx + first_idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, segments_in_block + 1, idx + first_idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_LT( first_idx + idx, last_idx, "" );
         const Index globalIdx = first_idx + idx;
         if constexpr( argumentCount< Function >() == 3 )
            function( first_segment_in_block + local_segmentIdx, globalIdx - shared_offsets[ local_segmentIdx ], globalIdx );
         else
            function( first_segment_in_block + local_segmentIdx, globalIdx );
      }
      idx += BlockSize;
   }
#endif
}

template< typename OffsetsView, typename Index, typename Function >
__global__
void
forElementsKernel( Index gridIdx, OffsetsView offsets, Index begin, Index end, Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   // We map one warp to each segment
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize();
   if( segmentIdx >= end )
      return;

   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   //const Index laneIdx = threadIdx.x % Backend::getWarpSize();  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += Backend::getWarpSize() ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
      localIdx += Backend::getWarpSize();
   }
#endif
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::forElements( IndexType begin, IndexType end, Function function ) const
{
   if( end <= begin )
      return;

   if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
      const Index segmentsCount = end - begin;
      std::size_t threadsCount;
      if constexpr( argumentCount< Function >() == 2 )  // we use scan kernel
         threadsCount = segmentsCount;
      else
         threadsCount = segmentsCount * Backend::getWarpSize();
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      dim3 blocksCount;
      dim3 gridsCount;
      Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
      for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
         Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
         if constexpr( argumentCount< Function >() == 3 ) {
            constexpr auto kernel = forElementsKernel< ConstOffsetsView, IndexType, Function >;
            Backend::launchKernelAsync( kernel, launch_config, gridIdx, this->offsets, begin, end, function );
         }
         else {
            constexpr auto kernel = forElementsBlockMergeKernel< ConstOffsetsView, IndexType, Function >;
            Backend::launchKernelAsync( kernel, launch_config, gridIdx, this->offsets, begin, end, function );
         }
      }
      Backend::streamSynchronize( launch_config.stream );
   }
   else {
      const auto offsetsView = this->offsets;
      // TODO: if constexpr could be just inside the lambda function l when nvcc allolws it
      if constexpr( argumentCount< Function >() == 3 ) {
         auto l = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
         {
            const IndexType begin = offsetsView[ segmentIdx ];
            const IndexType end = offsetsView[ segmentIdx + 1 ];
            IndexType localIdx( 0 );
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               function( segmentIdx, localIdx++, globalIdx );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
      else {
         auto l = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
         {
            const IndexType begin = offsetsView[ segmentIdx ];
            const IndexType end = offsetsView[ segmentIdx + 1 ];
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               function( segmentIdx, globalIdx );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::forAllElements( Function function ) const
{
   this->forElements( 0, this->getSegmentsCount(), function );
}

template< typename OffsetsView, typename ArrayView, typename Index, typename Function >
__global__
void
forElementsWithSegmentIndexesKernel( Index gridIdx,
                                     OffsetsView offsets,
                                     ArrayView segmentIndexes,
                                     Index begin,
                                     Index end,
                                     Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   // We map one warp to each segment
   const Index idx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize();
   if( idx >= end )
      return;
   TNL_ASSERT_GE( idx, 0, "" );
   TNL_ASSERT_LT( idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ idx ];
   TNL_ASSERT_GE( segmentIdx, 0, "Wrong index segment index - smaller that 0." );
   TNL_ASSERT_LT( segmentIdx, offsets.getSize() - 1, "Wrong index segment index - larger that the number of indexes." );

   const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   //const Index laneIdx = threadIdx.x % Backend::getWarpSize();  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += Backend::getWarpSize() ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      if constexpr( argumentCount< Function >() == 3 )
         function( segmentIdx, localIdx, globalIdx );
      else
         function( segmentIdx, globalIdx );
      localIdx += Backend::getWarpSize();
   }
#endif
}

template< typename OffsetsView, typename ArrayView, typename Index, typename Function, int SegmentsPerBlock, int BlockSize = 256 >
__global__
void
forElementsWithSegmentIndexesBlockMergeKernel( Index gridIdx,
                                               OffsetsView offsets,
                                               ArrayView segmentIndexes,
                                               const Index begin,
                                               const Index end,
                                               Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using CudaScan = Algorithms::detail::CudaBlockScanShfl< Algorithms::detail::ScanType::Exclusive, BlockSize, Plus, Index >;
   using ScanStorage = typename CudaScan::Storage;

   __shared__ ScanStorage scan_storage;
   __shared__ Index shared_offsets[ SegmentsPerBlock + 1 ];
   __shared__ Index shared_global_offsets[ SegmentsPerBlock ];
   __shared__ Index shared_segment_indexes[ SegmentsPerBlock ];

   const Index segmentIdx_ptr = begin + Backend::getGlobalBlockIdx_x( gridIdx ) * SegmentsPerBlock + threadIdx.x;
   const Index last_local_segment_idx = min( SegmentsPerBlock, end - begin - blockIdx.x * SegmentsPerBlock );
   if( segmentIdx_ptr < end && threadIdx.x < SegmentsPerBlock ) {
      shared_segment_indexes[ threadIdx.x ] = segmentIndexes[ segmentIdx_ptr ];
      shared_global_offsets[ threadIdx.x ] = offsets[ shared_segment_indexes[ threadIdx.x ] ];
   }

   Index value = 0;
   if( segmentIdx_ptr < end && threadIdx.x <= SegmentsPerBlock ) {
      const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
      TNL_ASSERT_GE( seg_idx, 0, "Wrong index of segment index - smaller that 0." );
      TNL_ASSERT_LT( seg_idx, offsets.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
      value = offsets[ seg_idx + 1 ] - offsets[ seg_idx ];
   }
   const Index v = CudaScan::scan( Plus{}, (Index) 0, value, threadIdx.x, scan_storage );
   if( threadIdx.x <= SegmentsPerBlock && threadIdx.x < BlockSize )
      shared_offsets[ threadIdx.x ] = v;

   // Compute the last offset in the block - this is necessary only SegmentsPerBlock == BlockSize
   if constexpr( SegmentsPerBlock == BlockSize )
      if( threadIdx.x == last_local_segment_idx - 1 ) {
         const Index seg_idx = segmentIndexes[ segmentIdx_ptr ];
         shared_offsets[ threadIdx.x + 1 ] = shared_offsets[ threadIdx.x ] + offsets[ seg_idx + 1 ] - offsets[ seg_idx ];
      }
   __syncthreads();
   const Index last_idx = shared_offsets[ last_local_segment_idx ];
   TNL_ASSERT_LT( last_idx, offsets[ offsets.getSize() - 1 ] - shared_segment_indexes[ 0 ], "" );

   /*if( threadIdx.x == 0 && blockIdx.x == 2 ) {
      Index i;
      for( i = 0; i < SegmentsPerBlock && begin + i < end; i++ ) {
         const Index seg_idx = segmentIndexes[ begin + i ];
         printf( "blockIdx %d: shared_segment_indexes[ %d] = %d shared_segment_sizes[ %d ] = %d shared_offsets[ %d ] = %d\n",
                 blockIdx.x,
                 i,
                 shared_segment_indexes[ i ],
                 seg_idx,
                 offsets[ seg_idx + 1 ] - offsets[ seg_idx ],
                 i,
                 shared_offsets[ i ] );
      }
      printf( "blockIdx %d: shared_offsets[ %d ] = %d\n", blockIdx.x, i, shared_offsets[ i ] );
      printf( "begin = %d end = %d last_local_segment_idx = %d last_idx = %d\n", begin, end, last_local_segment_idx, last_idx );
      //printf( "last_local_segment_idx = %d last_idx = %d\n", last_local_segment_idx, last_idx );
   }
   __syncthreads();*/

   Index idx = threadIdx.x;
   while( idx < last_idx ) {
      auto [ found, local_segmentIdx ] = Algorithms::findUpperBound( shared_offsets, last_local_segment_idx + 1, idx );
      if( found ) {
         local_segmentIdx--;
         TNL_ASSERT_GE( local_segmentIdx, 0, "" );
         TNL_ASSERT_LT( local_segmentIdx, last_local_segment_idx, "" );
         TNL_ASSERT_LT( local_segmentIdx, SegmentsPerBlock, "" );
         TNL_ASSERT_GE( shared_segment_indexes[ local_segmentIdx ], 0, "" );
         TNL_ASSERT_LT( shared_segment_indexes[ local_segmentIdx ], offsets.getSize() - 1, "" );

         const Index localIdx = idx - shared_offsets[ local_segmentIdx ];
         const Index globalIdx = shared_global_offsets[ local_segmentIdx ] + localIdx;
         TNL_ASSERT_GE( globalIdx, 0, "" );
         TNL_ASSERT_LT( globalIdx, offsets[ offsets.getSize() - 1 ], "" );
         if constexpr( argumentCount< Function >() == 3 )
            function( shared_segment_indexes[ local_segmentIdx ], localIdx, globalIdx );
         else
            function( shared_segment_indexes[ local_segmentIdx ], globalIdx );
      }
      idx += BlockSize;
   }

#endif
}

template< typename Device, typename Index >
template< typename Array, typename Function >
void
CSRBase< Device, Index >::forElements( const Array& segmentIndexes, Index begin, Index end, Function function ) const
{
   if( end <= begin )
      return;
   auto segmentIndexesView = segmentIndexes.getConstView();
   if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
      //std::cout << "Max offsets:" << this->offsets.getElement( this->offsets.getSize() - 1 ) << std::endl;
      const Index segmentsCount = end - begin;
      std::size_t threadsCount;
      //if constexpr( argumentCount< Function >() == 2 )  // we use scan kernel
      constexpr int ThreadsPerSegment = 4;
      constexpr int SegmentsPerBlock = 256 / ThreadsPerSegment;
      threadsCount = segmentsCount * ThreadsPerSegment;
      //else
      //threadsCount = segmentsCount * Backend::getWarpSize();
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      dim3 blocksCount;
      dim3 gridsCount;
      Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
      for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
         Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );

         /*constexpr auto kernel =
            forElementsWithSegmentIndexesKernel< ConstOffsetsView, typename Array::ConstViewType, IndexType, Function >;
         Backend::launchKernelAsync( kernel, launch_config, gridIdx, this->offsets, segmentIndexesView, begin, end, function
         );*/

         constexpr auto kernel = forElementsWithSegmentIndexesBlockMergeKernel< ConstOffsetsView,
                                                                                typename Array::ConstViewType,
                                                                                IndexType,
                                                                                Function,
                                                                                SegmentsPerBlock >;
         Backend::launchKernelAsync( kernel, launch_config, gridIdx, this->offsets, segmentIndexesView, begin, end, function );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
   else {
      const auto offsetsView = this->offsets;
      // TODO: if constexpr could be just inside the lambda function l when nvcc allolws it
      if constexpr( argumentCount< Function >() == 3 ) {
         auto l = [ = ] __cuda_callable__( IndexType idx ) mutable
         {
            TNL_ASSERT_LT( idx, segmentIndexesView.getSize(), "" );
            const IndexType segmentIdx = segmentIndexesView[ idx ];
            TNL_ASSERT_GE( segmentIdx, 0, "Wrong index of segment index - smaller that 0." );
            TNL_ASSERT_LT(
               segmentIdx, offsetsView.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
            const IndexType begin = offsetsView[ segmentIdx ];
            const IndexType end = offsetsView[ segmentIdx + 1 ];
            IndexType localIdx( 0 );
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
               TNL_ASSERT_LT( globalIdx, this->getStorageSize(), "" );
               function( segmentIdx, localIdx++, globalIdx );
            }
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
      else {  // argumentCount< Function >() == 2
         auto l = [ = ] __cuda_callable__( IndexType idx ) mutable
         {
            TNL_ASSERT_LT( idx, segmentIndexesView.getSize(), "" );
            const IndexType segmentIdx = segmentIndexesView[ idx ];
            TNL_ASSERT_GE( segmentIdx, 0, "Wrong index of segment index - smaller that 0." );
            TNL_ASSERT_LT(
               segmentIdx, offsetsView.getSize() - 1, "Wrong index of segment index - larger that the number of indexes." );
            const IndexType begin = offsetsView[ segmentIdx ];
            const IndexType end = offsetsView[ segmentIdx + 1 ];
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
               TNL_ASSERT_LT( globalIdx, this->getStorageSize(), "" );
               function( segmentIdx, globalIdx );
            }
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }
}

template< typename Device, typename Index >
template< typename Array, typename Function >
void
CSRBase< Device, Index >::forElements( const Array& segmentIndexes, Function function ) const
{
   this->forElements( segmentIndexes, 0, segmentIndexes.getSize(), function );
}

template< typename OffsetsView, typename Index, typename Condition, typename Function >
__global__
void
forElementsIfKernel( Index gridIdx, OffsetsView offsets, Index begin, Index end, Condition condition, Function function )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )

   // We map one warp to each segment
   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / Backend::getWarpSize();
   if( segmentIdx >= end || ! condition( segmentIdx ) )
      return;

   //const Index laneIdx = threadIdx.x & ( Backend::getWarpSize() - 1 );  // & is cheaper than %
   const Index laneIdx = threadIdx.x % Backend::getWarpSize();  // & is cheaper than %
   TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
   Index endIdx = offsets[ segmentIdx + 1 ];

   Index localIdx = laneIdx;
   for( Index globalIdx = offsets[ segmentIdx ] + laneIdx; globalIdx < endIdx; globalIdx += Backend::getWarpSize() ) {
      TNL_ASSERT_LT( globalIdx, endIdx, "" );
      /*printf( ">>> threadIdx %d segmentIdx %d laneIdx %d localIdx %d globalIdx %d \n",
              threadIdx.x,
              segmentIdx,
              laneIdx,
              localIdx,
              globalIdx );*/
      function( segmentIdx, localIdx, globalIdx );
      localIdx += Backend::getWarpSize();
   }
#endif
}

template< typename Device, typename Index >
template< typename Condition, typename Function >
void
CSRBase< Device, Index >::forElementsIf( IndexType begin, IndexType end, Condition condition, Function function ) const
{
   if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
      if( end <= begin )
         return;

      const Index warpsCount = end - begin;
      const std::size_t threadsCount = warpsCount * Backend::getWarpSize();
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      dim3 blocksCount;
      dim3 gridsCount;
      Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
      for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
         Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
         constexpr auto kernel = forElementsIfKernel< ConstOffsetsView, IndexType, Condition, Function >;
         Backend::launchKernelAsync( kernel, launch_config, gridIdx, this->offsets, begin, end, condition, function );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
   else {
      const auto offsetsView = this->offsets;
      auto l = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
      {
         const IndexType begin = offsetsView[ segmentIdx ];
         const IndexType end = offsetsView[ segmentIdx + 1 ];
         IndexType localIdx( 0 );
         if( condition( segmentIdx ) )
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               function( segmentIdx, localIdx++, globalIdx );
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }
}

template< typename Device, typename Index >
template< typename Condition, typename Function >
void
CSRBase< Device, Index >::forAllElementsIf( Condition condition, Function function ) const
{
   this->forElementsIf( 0, this->getSegmentsCount(), condition, function );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   const auto& self = *this;
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = self.getSegmentView( segmentIdx );
      function( segment );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::forAllSegments( Function&& function ) const
{
   this->forSegments( 0, this->getSegmentsCount(), function );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::sequentialForSegments( IndexType begin, IndexType end, Function&& function ) const
{
   for( IndexType i = begin; i < end; i++ )
      forSegments( i, i + 1, function );
}

template< typename Device, typename Index >
template< typename Function >
void
CSRBase< Device, Index >::sequentialForAllSegments( Function&& function ) const
{
   this->sequentialForSegments( 0, this->getSegmentsCount(), function );
}

}  // namespace TNL::Algorithms::Segments
