// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Backend.h>

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
template< typename Function >
void
CSRBase< Device, Index >::forElements( IndexType begin, IndexType end, Function function ) const
{
   if( end <= begin )
      return;

   if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
      const Index warpsCount = end - begin;
      const std::size_t threadsCount = warpsCount * Backend::getWarpSize();
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      dim3 blocksCount;
      dim3 gridsCount;
      Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
      for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
         Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
         constexpr auto kernel = forElementsKernel< ConstOffsetsView, IndexType, Function >;
         Backend::launchKernelAsync( kernel, launch_config, gridIdx, this->offsets, begin, end, function );
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
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
            function( segmentIdx, localIdx++, globalIdx );
      };
      Algorithms::parallelFor< Device >( begin, end, l );
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
   TNL_ASSERT_GE( 0, segmentIndexes.getSize(), "" );
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
      function( segmentIdx, localIdx, globalIdx );
      localIdx += Backend::getWarpSize();
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
      const Index warpsCount = end - begin;
      const std::size_t threadsCount = warpsCount * Backend::getWarpSize();
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      dim3 blocksCount;
      dim3 gridsCount;
      Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
      for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
         Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
         constexpr auto kernel =
            forElementsWithSegmentIndexesKernel< ConstOffsetsView, typename Array::ConstViewType, IndexType, Function >;
         Backend::launchKernelAsync( kernel, launch_config, gridIdx, this->offsets, segmentIndexesView, begin, end, function );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
   else {
      const auto offsetsView = this->offsets;
      auto l = [ = ] __cuda_callable__( IndexType idx ) mutable
      {
         TNL_ASSERT_LT( idx, segmentIndexesView.getSize(), "" );
         const IndexType segmentIdx = segmentIndexesView[ idx ];
         TNL_ASSERT_GE( segmentIdx, 0, "Wrong index segment index - smaller that 0." );
         TNL_ASSERT_LT(
            segmentIdx, offsetsView.getSize() - 1, "Wrong index segment index - larger that the number of indexes." );
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
