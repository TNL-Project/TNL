// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include "TraversingKernels_Ellpack.h"
#include "TraversingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
struct TraversingOperations< EllpackView< Device, Index, Organization, Alignment > >
: public TraversingOperationsBase< EllpackView< Device, Index, Organization, Alignment > >
{
   using ViewType = EllpackView< Device, Index, Organization, Alignment >;
   // ViewType is the same as ConstViewType for Ellpack !!!!!
   //using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = typename std::remove_const< Index >::type;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElementsSequential( const ViewType& segments,
                          IndexBegin begin,
                          IndexEnd end,
                          Function&& function,
                          const LaunchConfiguration& launchConfig )
   {
      const IndexType segmentSize = segments.getSegmentSize();
      const IndexType storageSize = segments.getStorageSize();
      const IndexType alignedSize = segments.getAlignedSize();
      auto l = [ segmentSize, storageSize, alignedSize, function ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         if constexpr( Organization == RowMajorOrder ) {
            const IndexType begin = segmentIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, localIdx++, globalIdx );
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            }
         }
         else {
            const IndexType begin = segmentIdx;
            const IndexType end = storageSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, localIdx++, globalIdx );
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, globalIdx );
            }
         }
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                LaunchConfiguration launchConfig )  // TODO: Function&& does not work here - why???
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
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed )
               threadsCount *= (std::size_t) launchConfig.getThreadsPerSegmentCount();
            else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMerged )
               threadsCount *= (std::size_t) segments.getSegmentSize();

            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            const IndexType totalThreadsCount = blocksCount.x * launchConfig.blockSize.x;
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  constexpr auto kernel =
                     forElementsKernel_Ellpack< ViewType, IndexType, std::remove_reference_t< Function >, Organization >;
                  Backend::launchKernelAsync( kernel,
                                              launchConfig,
                                              gridIdx,
                                              totalThreadsCount,
                                              launchConfig.getThreadsPerSegmentCount(),
                                              segments,
                                              begin,
                                              end,
                                              function );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMerged ) {
                  constexpr auto kernel = forElementsBlockMergeKernel_Ellpack< ViewType,
                                                                               IndexType,
                                                                               std::remove_reference_t< Function >,
                                                                               Organization,
                                                                               256 >;
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        Backend::launchKernelAsync( kernel, launchConfig, gridIdx, segments, begin, end, function );
                        break;
                     default:
                        throw std::invalid_argument( "Unsupported threads per segment ( "
                                                     + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                                     + " ) count for Ellpack segments." );
                        break;
                  }
               }
               else
                  throw std::invalid_argument( "Unsupported threads to segments mapping for Ellpack segments." );
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
   forElementsSequential( const ViewType& segments,
                          const Array& segmentIndexes,
                          Function&& function,
                          const LaunchConfiguration& launchConfig )
   {
      auto segmentIndexesView = segmentIndexes.getConstView();
      const IndexType segmentSize = segments.getSegmentSize();
      const IndexType storageSize = segments.getStorageSize();
      const IndexType alignedSize = segments.getAlignedSize();
      auto l = [ segmentSize, storageSize, alignedSize, segmentIndexesView, function ] __cuda_callable__(
                  const IndexType idx ) mutable
      {
         const IndexType segmentIdx = segmentIndexesView[ idx ];
         if constexpr( Organization == RowMajorOrder ) {
            const IndexType begin = segmentIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, localIdx++, globalIdx );
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            }
         }
         else {
            const IndexType begin = segmentIdx;
            const IndexType end = storageSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, localIdx++, globalIdx );
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, globalIdx );
            }
         }
      };
      Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
   }

   template< typename Array, typename Function >
   static void
   forElements( const ViewType& segments, const Array& segmentIndexes, Function&& function, LaunchConfiguration launchConfig )
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
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed )
               threadsCount *= (std::size_t) launchConfig.getThreadsPerSegmentCount();
            else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMerged )
               threadsCount *= (std::size_t) segments.getSegmentSize();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            const IndexType totalThreadsCount = blocksCount.x * launchConfig.blockSize.x;
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  constexpr auto kernel = forElementsWithSegmentIndexesKernel_Ellpack< ViewType,
                                                                                       typename Array::ConstViewType,
                                                                                       IndexType,
                                                                                       std::remove_reference_t< Function >,
                                                                                       Organization >;
                  Backend::launchKernelAsync( kernel,
                                              launchConfig,
                                              gridIdx,
                                              totalThreadsCount,
                                              launchConfig.getThreadsPerSegmentCount(),
                                              segments,
                                              segmentIndexesView,
                                              function );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::BlockMerged ) {
                  constexpr auto kernel =
                     forElementsWithSegmentIndexesBlockMergeKernel_Ellpack< ViewType,
                                                                            typename Array::ConstViewType,
                                                                            IndexType,
                                                                            std::remove_reference_t< Function >,
                                                                            Organization,
                                                                            256,    // SegmentsPerBlock
                                                                            256 >;  // BlockSize

                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        Backend::launchKernelAsync( kernel, launchConfig, gridIdx, segments, segmentIndexesView, function );
                        break;
                     default:
                        throw std::invalid_argument( "Unsupported threads per segment ( "
                                                     + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                                     + " ) count for Ellpack segments." );
                        break;
                  }
               }
               else
                  throw std::invalid_argument( "Unsupported threads to segments mapping for Ellpack segments." );
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
   forElementsIfSequential( const ViewType& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Condition condition,
                            Function&& function,
                            const LaunchConfiguration& launchConfig )
   {
      const IndexType segmentSize = segments.getSegmentSize();
      const IndexType storageSize = segments.getStorageSize();
      const IndexType alignedSize = segments.getAlignedSize();
      auto l =
         [ segmentSize, storageSize, alignedSize, condition, function ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         if( ! condition( segmentIdx ) )
            return;
         if constexpr( Organization == RowMajorOrder ) {
            const IndexType begin = segmentIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, localIdx++, globalIdx );
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            }
         }
         else {
            const IndexType begin = segmentIdx;
            const IndexType end = storageSize;
            if constexpr( argumentCount< Function >() == 3 ) {
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, localIdx++, globalIdx );
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, globalIdx );
            }
         }
      };
      Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ViewType& segments,
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
            const IndexType totalThreadsCount = blocksCount.x * launch_config.blockSize.x;
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );

               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  constexpr auto kernel = forElementsIfKernel_Ellpack< ViewType,
                                                                       IndexType,
                                                                       std::remove_reference_t< Condition >,
                                                                       std::remove_reference_t< Function >,
                                                                       Organization >;
                  Backend::launchKernelAsync( kernel,
                                              launch_config,
                                              gridIdx,
                                              totalThreadsCount,
                                              launchConfig.getThreadsPerSegmentCount(),
                                              segments,
                                              begin,
                                              end,
                                              condition,
                                              function );
               }
               else {  // BlockMerge mapping - this mapping is currently the default one
                  constexpr auto kernel = forElementsIfBlockMergeKernel_Ellpack< ViewType,
                                                                                 IndexType,
                                                                                 std::remove_reference_t< Condition >,
                                                                                 std::remove_reference_t< Function >,
                                                                                 Organization,
                                                                                 256,
                                                                                 256 >;
                  Backend::launchKernelAsync( kernel, launch_config, gridIdx, segments, begin, end, condition, function );
               }
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
