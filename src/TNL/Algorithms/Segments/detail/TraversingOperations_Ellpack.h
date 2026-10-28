// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include "TraversingKernels_Ellpack_Uniform.h"
#include "TraversingKernels_Ellpack_PerElement.h"
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
   using IndexType = std::remove_const_t< Index >;

   /**
    * \brief Resolves the `Default` mapping of `launchConfig` in place
    * and rejects the mappings without a GPU traversal kernel.
    *
    * The `Default` mapping resolves to the `Uniform` mapping with one thread per segment.
    */
   static void
   resolveLaunchConfiguration( LaunchConfiguration& launchConfig )
   {
      switch( launchConfig.getThreadsToSegmentsMapping() ) {
         case ThreadsToSegmentsMapping::Default:
            launchConfig.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::Uniform );
            launchConfig.setThreadsPerSegmentCount( 1 );
            break;
         case ThreadsToSegmentsMapping::Uniform:
         case ThreadsToSegmentsMapping::PerElement:
            break;
         default:
            throw std::invalid_argument( "Unsupported threads to segments mapping for the traversal of Ellpack segments." );
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElementsSequential(
      const ViewType& segments,
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
         (void) segmentSize;  // To suppress unused variable warning
         (void) storageSize;
         (void) alignedSize;
         if constexpr( Organization == RowMajorOrder ) {
            const IndexType begin = segmentIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            if constexpr( callableArgumentCount< Function >() == 3 ) {
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
            if constexpr( callableArgumentCount< Function >() == 3 ) {
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
   forElements(
      const ViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;

      if( launchConfig.blockSize.x == 1 )
         launchConfig.blockSize.x = 256;

      if constexpr( std::is_same_v< DeviceType, Devices::GPU > ) {
         resolveLaunchConfiguration( launchConfig );
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
             && launchConfig.getThreadsPerSegmentCount() == 1 )
         {
            forElementsSequential( segments, begin, end, std::forward< Function >( function ), launchConfig );
         }
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::PerElement )
               threadsCount *= static_cast< std::size_t >( segments.getSegmentSize() );

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            const std::size_t totalThreadsCount = static_cast< std::size_t >( blocksCount.x ) * launchConfig.blockSize.x;
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform ) {
                  auto condition = [] __cuda_callable__( IndexType )
                  {
                     return true;
                  };
                  constexpr auto kernel = forElements_Ellpack_Uniform_If<
                     ViewType,
                     IndexType,
                     decltype( condition ),
                     std::remove_reference_t< Function >,
                     Organization >;
                  Backend::launchKernelAsync(
                     kernel,
                     launchConfig,
                     gridIdx,
                     totalThreadsCount,
                     launchConfig.getThreadsPerSegmentCount(),
                     segments,
                     begin,
                     end,
                     condition,
                     function );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::PerElement ) {
                  constexpr auto kernel = forElements_Ellpack_PerElement<
                     ViewType,
                     IndexType,
                     std::remove_reference_t< Function >,
                     Organization,
                     256 >;
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 1:
                        Backend::launchKernelAsync( kernel, launchConfig, gridIdx, segments, begin, end, function );
                        break;
                     default:
                        throw std::invalid_argument(
                           "Unsupported threads per segment ( " + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                           + " ) count for Ellpack segments." );
                  }
               }
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
   forElementsSequential(
      const ViewType& segments,
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
         (void) segmentSize;  // To suppress unused variable warning
         (void) storageSize;
         (void) alignedSize;
         const IndexType segmentIdx = segmentIndexesView[ idx ];
         if constexpr( Organization == RowMajorOrder ) {
            const IndexType begin = segmentIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            if constexpr( callableArgumentCount< Function >() == 3 ) {
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
            if constexpr( callableArgumentCount< Function >() == 3 ) {
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

      if constexpr( std::is_same_v< DeviceType, Devices::GPU > ) {
         resolveLaunchConfiguration( launchConfig );
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
             && launchConfig.getThreadsPerSegmentCount() == 1 )
         {
            forElementsSequential( segments, segmentIndexes, std::forward< Function >( function ), launchConfig );
         }
         else {
            auto segmentIndexesView = segmentIndexes.getConstView();
            std::size_t threadsCount = segmentIndexes.getSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::PerElement )
               threadsCount *= static_cast< std::size_t >( segments.getSegmentSize() );

            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launchConfig.blockSize, blocksCount, gridsCount, threadsCount );
            const std::size_t totalThreadsCount = static_cast< std::size_t >( blocksCount.x ) * launchConfig.blockSize.x;
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform ) {
                  constexpr auto kernel = forElements_Ellpack_Uniform_WithIndexes<
                     ViewType,
                     typename Array::ConstViewType,
                     IndexType,
                     std::remove_reference_t< Function >,
                     Organization >;
                  Backend::launchKernelAsync(
                     kernel,
                     launchConfig,
                     gridIdx,
                     totalThreadsCount,
                     launchConfig.getThreadsPerSegmentCount(),
                     segments,
                     segmentIndexesView,
                     function );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::PerElement ) {
                  constexpr auto kernel = forElements_Ellpack_PerElement_WithIndexes<
                     ViewType,
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
                        throw std::invalid_argument(
                           "Unsupported threads per segment ( " + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                           + " ) count for Ellpack segments." );
                  }
               }
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
   forElementsIfSequential(
      const ViewType& segments,
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
         (void) segmentSize;  // To suppress unused variable warning
         (void) storageSize;
         (void) alignedSize;
         if( ! condition( segmentIdx ) )
            return;
         if constexpr( Organization == RowMajorOrder ) {
            const IndexType begin = segmentIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            if constexpr( callableArgumentCount< Function >() == 3 ) {
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
            if constexpr( callableArgumentCount< Function >() == 3 ) {
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
   forElementsIf(
      const ViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      LaunchConfiguration launchConfig )
   {
      if constexpr( std::is_same_v< Device, Devices::GPU > ) {
         if( end <= begin )
            return;

         resolveLaunchConfiguration( launchConfig );
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
             && launchConfig.getThreadsPerSegmentCount() == 1 )
         {
            forElementsIfSequential( segments, begin, end, std::forward< Condition >( condition ), function, launchConfig );
         }
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );

            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            const std::size_t totalThreadsCount = static_cast< std::size_t >( blocksCount.x ) * launch_config.blockSize.x;
            for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );

               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform ) {
                  constexpr auto kernel = forElements_Ellpack_Uniform_If<
                     ViewType,
                     IndexType,
                     std::remove_reference_t< Condition >,
                     std::remove_reference_t< Function >,
                     Organization >;
                  Backend::launchKernelAsync(
                     kernel,
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
               else {  // PerElement mapping - this mapping is currently the default one
                  constexpr auto kernel = forElements_Ellpack_PerElement_If<
                     ViewType,
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

}  // namespace TNL::Algorithms::Segments::detail
