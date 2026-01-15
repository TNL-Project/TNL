// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "TraversingKernels_SlicedEllpack.h"
#include "TraversingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
struct TraversingOperations< SlicedEllpackView< Device, Index, Organization, SliceSize > >
: public TraversingOperationsBase< SlicedEllpackView< Device, Index, Organization, SliceSize > >
{
   using ViewType = SlicedEllpackView< Device, Index, Organization, SliceSize >;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = typename std::remove_const< Index >::type;
   using ConstOffsetsView = typename ViewType::ConstOffsetsView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElementsSequential( const ConstViewType& segments,
                          IndexBegin begin,
                          IndexEnd end,
                          Function&& function,
                          const LaunchConfiguration& launchConfig )
   {
      const auto sliceSegmentSizes_view = segments.getSliceSegmentSizesView();
      const auto sliceOffsets_view = segments.getSliceOffsetsView();
      auto l = [ sliceSegmentSizes_view, sliceOffsets_view, function ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         (void) sliceSegmentSizes_view;  // To suppress unused variable warning
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         if constexpr( Organization == RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
            const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            }
         }
         else {  // ColumnMajorOrder
            const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize )
                  function( segmentIdx, globalIdx );
            }
         }
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Function,
             typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
   static void
   forElements( const ConstViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;

      if( launchConfig.blockSize.x == 1 )
         launchConfig.blockSize.x = 256;

      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > || std::is_same_v< DeviceType, Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
             && launchConfig.getThreadsPerSegmentCount() == 1 )
            forElementsSequential( segments, begin, end, std::forward< Function >( function ), launchConfig );
         else {
            std::size_t threadsCount( 0 );
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
               const std::size_t segmentsCount = end - begin;
               const std::size_t slicesCount = roundUpDivision( segmentsCount, SliceSize );
               threadsCount = slicesCount * (std::size_t) SliceSize * (std::size_t) launchConfig.getThreadsPerSegmentCount();
               if( threadsCount > std::numeric_limits< IndexType >::max() )
                  throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );
            }
            else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMerged ) {
               const IndexType firstSegmentIdx = ( begin / SliceSize ) * SliceSize;
               const IndexType lastSegmentIdx = roundUpDivision( end, SliceSize ) * SliceSize;
               const IndexType segmentsCount = lastSegmentIdx - firstSegmentIdx;
               threadsCount = segmentsCount / launchConfig.getThreadsPerSegmentCount();
               launchConfig.blockSize.x = 256;
            }

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  constexpr auto kernel = forElementsKernel_SlicedEllpack< ConstViewType,
                                                                           IndexType,
                                                                           std::remove_reference_t< Function >,
                                                                           Organization,
                                                                           SliceSize >;
                  Backend::launchKernelAsync(
                     kernel, launchConfig, gridIdx, launchConfig.getThreadsPerSegmentCount(), segments, begin, end, function );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMerged ) {
                  constexpr auto kernel = forElementsBlockMergeKernel_SlicedEllpack< ConstViewType,
                                                                                     IndexType,
                                                                                     std::remove_reference_t< Function >,
                                                                                     Organization,
                                                                                     SliceSize,
                                                                                     256 >;
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        Backend::launchKernelAsync( kernel, launchConfig, gridIdx, segments, begin, end, function );
                        break;
                     default:
                        throw std::invalid_argument( "Unsupported threads per segment ( "
                                                     + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                                     + " ) count for Sliced Ellpack segments." );
                        break;
                  }
               }
               else
                  throw std::invalid_argument( "Unsupported threads to segments mapping for Sliced Ellpack segments." );
            }
            Backend::streamSynchronize( launchConfig.stream );
         }
      }
      else {
         forElementsSequential( segments, begin, end, std::forward< Function >( function ), launchConfig );
      }
   }

   template< typename Array, typename Function >
   static void
   forElementsSequential( const ConstViewType& segments,
                          const Array& segmentIndexes,
                          Function&& function,
                          const LaunchConfiguration& launchConfig )
   {
      auto segmentIndexes_view = segmentIndexes.getConstView();
      const auto sliceSegmentSizes_view = segments.getSliceSegmentSizesView();
      const auto sliceOffsets_view = segments.getSliceOffsetsView();
      auto l = [ segmentIndexes_view, sliceSegmentSizes_view, sliceOffsets_view, function ] __cuda_callable__(
                  const IndexType idx ) mutable
      {
         (void) sliceSegmentSizes_view;  // To suppress unused variable warning
         const IndexType segmentIdx = segmentIndexes_view[ idx ];
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         if constexpr( Organization == RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
            const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            }
         }
         else {
            const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize )
                  function( segmentIdx, globalIdx );
            }
         }
      };
      Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
   }

   template< typename Array, typename Function >
   static void
   forElements( const ConstViewType& segments,
                const Array& segmentIndexes,
                Function&& function,
                LaunchConfiguration launchConfig )
   {
      if( launchConfig.blockSize.x == 1 )
         launchConfig.blockSize.x = 256;

      if constexpr( std::is_same_v< DeviceType, Devices::Cuda > || std::is_same_v< DeviceType, Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
             && launchConfig.getThreadsPerSegmentCount() == 1 )
            forElementsSequential( segments, segmentIndexes, std::forward< Function >( function ), launchConfig );
         else {
            auto segmentIndexesView = segmentIndexes.getConstView();
            std::size_t threadsCount = segmentIndexes.getSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
               threadsCount *= (std::size_t) launchConfig.getThreadsPerSegmentCount();
            }
            else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMerged )
               launchConfig.blockSize.x = 256;

            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  constexpr auto kernel =
                     forElementsWithSegmentIndexesKernel_SlicedEllpack< ConstViewType,
                                                                        typename Array::ConstViewType,
                                                                        IndexType,
                                                                        std::remove_reference_t< Function >,
                                                                        Organization,
                                                                        SliceSize >;
                  Backend::launchKernelAsync( kernel,
                                              launchConfig,
                                              gridIdx,
                                              launchConfig.getThreadsPerSegmentCount(),
                                              segments,
                                              segmentIndexesView,
                                              function );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMerged ) {
                  constexpr auto kernel =
                     forElementsWithSegmentIndexesBlockMergeKernel_SlicedEllpack< ConstViewType,
                                                                                  typename Array::ConstViewType,
                                                                                  IndexType,
                                                                                  std::remove_reference_t< Function >,
                                                                                  Organization,
                                                                                  SliceSize,
                                                                                  256,    // SegmentsPerBlock
                                                                                  256 >;  // BlockSize
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        Backend::launchKernelAsync( kernel, launchConfig, gridIdx, segments, segmentIndexesView, function );
                        break;
                     default:
                        throw std::invalid_argument( "Unsupported threads per segment ( "
                                                     + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                                     + " ) count for Sliced Ellpack segments." );
                        break;
                  }
               }
               else
                  throw std::invalid_argument( "Unsupported threads to segments mapping for Sliced Ellpack segments." );
            }
            Backend::streamSynchronize( launchConfig.stream );
         }
      }
      else {
         forElementsSequential( segments, segmentIndexes, std::forward< Function >( function ), launchConfig );
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIfSequential( const ConstViewType& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Condition&& condition,
                            Function&& function,
                            const LaunchConfiguration& launchConfig )
   {
      const auto sliceSegmentSizes_view = segments.getSliceSegmentSizesView();
      const auto sliceOffsets_view = segments.getSliceOffsetsView();
      auto l = [ sliceSegmentSizes_view, sliceOffsets_view, condition, function ] __cuda_callable__(
                  const IndexType segmentIdx ) mutable
      {
         (void) sliceSegmentSizes_view;  // To suppress unused variable warning
         if( ! condition( segmentIdx ) )
            return;
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         if constexpr( Organization == RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
            const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            }
         }
         else {
            const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize ) {
                  // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
                  function( segmentIdx, localIdx, globalIdx );
                  localIdx++;
#else
                  function( segmentIdx, localIdx++, globalIdx );
#endif
               }
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize )
                  function( segmentIdx, globalIdx );
            }
         }
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ConstViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Function&& function,
                  LaunchConfiguration launchConfig )
   {
      if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
         if( end <= begin )
            return;

         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
             && launchConfig.getThreadsPerSegmentCount() == 1 )
            forElementsIfSequential( segments, begin, end, std::forward< Condition >( condition ), function, launchConfig );
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed )
               threadsCount *= (std::size_t) launchConfig.getThreadsPerSegmentCount();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );

            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );

               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  constexpr auto kernel = forElementsIfKernel_SlicedEllpack< ConstViewType,
                                                                             IndexType,
                                                                             std::remove_reference_t< Condition >,
                                                                             std::remove_reference_t< Function >,
                                                                             Organization,
                                                                             SliceSize >;
                  Backend::launchKernelAsync( kernel,
                                              launch_config,
                                              gridIdx,
                                              launchConfig.getThreadsPerSegmentCount(),
                                              segments,
                                              begin,
                                              end,
                                              condition,
                                              function );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMerged ) {
                  constexpr auto kernel = forElementsIfBlockMergeKernel_SlicedEllpack< ConstViewType,
                                                                                       IndexType,
                                                                                       std::remove_reference_t< Condition >,
                                                                                       std::remove_reference_t< Function >,
                                                                                       Organization,
                                                                                       SliceSize,
                                                                                       256,
                                                                                       256 >;
                  Backend::launchKernelAsync( kernel, launch_config, gridIdx, segments, begin, end, condition, function );
               }
               else
                  throw std::invalid_argument( "Unsupported threads to segments mapping for Sliced Ellpack segments." );
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else {
         forElementsIfSequential( segments, begin, end, std::forward< Condition >( condition ), function, launchConfig );
      }
   }
};

}  //namespace TNL::Algorithms::Segments::detail
