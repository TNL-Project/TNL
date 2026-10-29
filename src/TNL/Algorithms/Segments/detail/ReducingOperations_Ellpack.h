// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include "LaunchConfiguration.h"
#include "FetchLambdaAdapter.h"
#include "ReducingKernels_Ellpack_Uniform.h"
#include "ReducingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
struct ReducingOperations< EllpackView< Device, Index, Organization, Alignment > >
: public ReducingOperationsBase< EllpackView< Device, Index, Organization, Alignment > >
{
   using SegmentsViewType = EllpackView< Device, Index, Organization, Alignment >;
   using ConstViewType = typename SegmentsViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;

   /**
    * \brief Resolves the `Default` mapping of `launchConfig` in place
    * and rejects the mappings without a GPU reduction kernel.
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
            break;
         default:
            throw std::invalid_argument( "Unsupported threads to segments mapping for the reduction of Ellpack segments." );
      }
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType,
      typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
   static void
   reduceSegments(
      const ConstViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::GPU > ) {
            resolveLaunchConfiguration( launchConfig );
            const int warpSize = Backend::getWarpSize( Backend::getDevice() );
            if( end <= begin )
               return;
            const std::size_t threadsCount = static_cast< std::size_t >( end - begin )
                                           * static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            auto launch = [ & ]( auto kernel )
            {
               for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
                  Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
                  Backend::launchKernelSync(
                     kernel, launch_config, gridIdx, segments, begin, end, fetch, reduction, storer, identity );
               }
            };
            dispatchThreadsPerSegment< 1, 2, 4, 8, 16, 32, 64 >(
               launchConfig.getThreadsPerSegmentCount(),
               [ & ]( auto threadsPerSegment )
               {
                  constexpr int TPS = decltype( threadsPerSegment )::value;
                  if constexpr( TPS <= Backend::getMaxWarpSize() ) {
                     if( TPS <= warpSize ) {
                        launch(
                           reduceSegments_Ellpack_Uniform<
                              TPS,
                              ConstViewType,
                              IndexBegin,
                              IndexEnd,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              ReturnType > );
                        return;
                     }
                  }
                  throw std::invalid_argument(
                     "Unsupported threads per segment count ( " + std::to_string( TPS )
                     + " ) for Ellpack segments on GPU with warp size " + std::to_string( warpSize ) + "." );
               },
               "Ellpack" );
         }
         else {  // CPU
            const IndexType segmentSize = segments.getSegmentSize();
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               ReturnType result = identity;
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++ )
                  result = reduction(
                     result, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
               storer( segmentIdx, result );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {  // ColumnMajorOrder
         // There is a single kernel for the column-major organization,
         // so the launch configuration is only checked for an unsupported mapping.
         if constexpr( std::is_same_v< Device, Devices::GPU > )
            resolveLaunchConfiguration( launchConfig );
         const IndexType storageSize = segments.getStorageSize();
         const IndexType alignedSize = segments.getAlignedSize();
         auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
         {
            const IndexType begin = segmentIdx;
            const IndexType end = storageSize;
            ReturnType aux = identity;
            IndexType localIdx = 0;
            for( IndexType j = begin; j < end; j += alignedSize )
               aux = reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
            storer( segmentIdx, aux );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }

   template<
      typename Array,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexes(
      const ConstViewType& segments,
      const Array& segmentIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      using ArrayView = typename Array::ConstViewType;
      auto segmentIndexes_view = segmentIndexes.getConstView();
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::GPU > ) {
            resolveLaunchConfiguration( launchConfig );
            const int warpSize = Backend::getWarpSize( Backend::getDevice() );
            if( segmentIndexes.getSize() == 0 )
               return;
            const std::size_t threadsCount = static_cast< std::size_t >( segmentIndexes.getSize() )
                                           * static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            auto launch = [ & ]( auto kernel )
            {
               for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
                  Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
                  Backend::launchKernelSync(
                     kernel,
                     launch_config,
                     gridIdx,
                     segments,
                     segmentIndexes.getConstView(),
                     fetch,
                     reduction,
                     storer,
                     identity );
               }
            };
            dispatchThreadsPerSegment< 1, 2, 4, 8, 16, 32, 64 >(
               launchConfig.getThreadsPerSegmentCount(),
               [ & ]( auto threadsPerSegment )
               {
                  constexpr int TPS = decltype( threadsPerSegment )::value;
                  if constexpr( TPS <= Backend::getMaxWarpSize() ) {
                     if( TPS <= warpSize ) {
                        launch(
                           reduceSegments_Ellpack_Uniform_WithIndexes<
                              TPS,
                              ConstViewType,
                              ArrayView,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              ReturnType > );
                        return;
                     }
                  }
                  throw std::invalid_argument(
                     "Unsupported threads per segment count ( " + std::to_string( TPS )
                     + " ) for Ellpack segments on GPU with warp size " + std::to_string( warpSize ) + "." );
               },
               "Ellpack" );
         }
         else {  // CPU
            const IndexType segmentSize = segments.getSegmentSize();
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx_idx ) mutable
            {
               TNL_ASSERT_LE( segmentIdx_idx, segmentIndexes_view.getSize(), "" );
               const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               ReturnType result = identity;
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++ )
                  result = reduction(
                     result, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
               storer( segmentIdx_idx, segmentIdx, result );
            };
            Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
         }
      }
      else {  // ColumnMajorOrder
         // There is a single kernel for the column-major organization,
         // so the launch configuration is only checked for an unsupported mapping.
         if constexpr( std::is_same_v< Device, Devices::GPU > )
            resolveLaunchConfiguration( launchConfig );
         const IndexType storageSize = segments.getStorageSize();
         const IndexType alignedSize = segments.getAlignedSize();
         auto l = [ = ] __cuda_callable__( const IndexType segmentIdx_idx ) mutable
         {
            TNL_ASSERT_LE( segmentIdx_idx, segmentIndexes_view.getSize(), "" );
            const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
            const IndexType begin = segmentIdx;
            const IndexType end = storageSize;
            ReturnType result = identity;
            IndexType localIdx = 0;
            for( IndexType j = begin; j < end; j += alignedSize )
               result =
                  reduction( result, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
            storer( segmentIdx_idx, segmentIdx, result );
         };
         Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
      }
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument(
      const ConstViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::GPU > ) {
            resolveLaunchConfiguration( launchConfig );
            const int warpSize = Backend::getWarpSize( Backend::getDevice() );
            if( end <= begin )
               return;
            const std::size_t threadsCount = static_cast< std::size_t >( end - begin )
                                           * static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            auto launch = [ & ]( auto kernel )
            {
               for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
                  Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
                  Backend::launchKernelSync(
                     kernel, launch_config, gridIdx, segments, begin, end, fetch, reduction, storer, identity );
               }
            };
            dispatchThreadsPerSegment< 1, 2, 4, 8, 16, 32, 64 >(
               launchConfig.getThreadsPerSegmentCount(),
               [ & ]( auto threadsPerSegment )
               {
                  constexpr int TPS = decltype( threadsPerSegment )::value;
                  if constexpr( TPS <= Backend::getMaxWarpSize() ) {
                     if( TPS <= warpSize ) {
                        launch(
                           reduceSegments_Ellpack_Uniform_WithArgument<
                              TPS,
                              ConstViewType,
                              IndexBegin,
                              IndexEnd,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              ReturnType > );
                        return;
                     }
                  }
                  throw std::invalid_argument(
                     "Unsupported threads per segment count ( " + std::to_string( TPS )
                     + " ) for Ellpack segments on GPU with warp size " + std::to_string( warpSize ) + "." );
               },
               "Ellpack" );
         }
         else {  // CPU
            const IndexType segmentSize = segments.getSegmentSize();
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               ReturnType result = identity;
               IndexType argument = 0;
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++, localIdx++ )
                  reduction(
                     result,
                     detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, j ),
                     argument,
                     localIdx );
               bool emptySegment = ( segmentSize == 0 );
               storer( segmentIdx, argument, result, emptySegment );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {  // ColumnMajorOrder
         // There is a single kernel for the column-major organization,
         // so the launch configuration is only checked for an unsupported mapping.
         if constexpr( std::is_same_v< Device, Devices::GPU > )
            resolveLaunchConfiguration( launchConfig );
         const IndexType storageSize = segments.getStorageSize();
         const IndexType alignedSize = segments.getAlignedSize();
         auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
         {
            const IndexType begin = segmentIdx;
            const IndexType end = storageSize;
            ReturnType result = identity;
            IndexType argument = 0;
            IndexType localIdx = 0;
            for( IndexType j = begin; j < end; j += alignedSize, localIdx++ )
               reduction(
                  result,
                  detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, j ),
                  argument,
                  localIdx );
            bool emptySegment = ( segments.getSegmentSize() == 0 );
            storer( segmentIdx, argument, result, emptySegment );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }

   template<
      typename Array,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexesAndArgument(
      const ConstViewType& segments,
      const Array& segmentIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      using ArrayView = typename Array::ConstViewType;
      auto segmentIndexes_view = segmentIndexes.getConstView();
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::GPU > ) {
            resolveLaunchConfiguration( launchConfig );
            const int warpSize = Backend::getWarpSize( Backend::getDevice() );
            if( segmentIndexes.getSize() == 0 )
               return;
            const std::size_t threadsCount = static_cast< std::size_t >( segmentIndexes.getSize() )
                                           * static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            auto launch = [ & ]( auto kernel )
            {
               for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx++ ) {
                  Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
                  Backend::launchKernelSync(
                     kernel,
                     launch_config,
                     gridIdx,
                     segments,
                     segmentIndexes.getConstView(),
                     fetch,
                     reduction,
                     storer,
                     identity );
               }
            };
            dispatchThreadsPerSegment< 1, 2, 4, 8, 16, 32, 64 >(
               launchConfig.getThreadsPerSegmentCount(),
               [ & ]( auto threadsPerSegment )
               {
                  constexpr int TPS = decltype( threadsPerSegment )::value;
                  if constexpr( TPS <= Backend::getMaxWarpSize() ) {
                     if( TPS <= warpSize ) {
                        launch(
                           reduceSegments_Ellpack_Uniform_WithIndexesAndArgument<
                              TPS,
                              ConstViewType,
                              ArrayView,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              ReturnType > );
                        return;
                     }
                  }
                  throw std::invalid_argument(
                     "Unsupported threads per segment count ( " + std::to_string( TPS )
                     + " ) for Ellpack segments on GPU with warp size " + std::to_string( warpSize ) + "." );
               },
               "Ellpack" );
         }
         else {  // CPU
            const IndexType segmentSize = segments.getSegmentSize();
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx_idx ) mutable
            {
               TNL_ASSERT_LE( segmentIdx_idx, segmentIndexes_view.getSize(), "" );
               const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               ReturnType result = identity;
               IndexType argument = 0;
               IndexType localIdx = 0;
               for( IndexType j = begin; j < end; j++, localIdx++ )
                  reduction(
                     result,
                     detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, j ),
                     argument,
                     localIdx );
               bool emptySegment = ( segmentSize == 0 );
               storer( segmentIdx_idx, segmentIdx, argument, result, emptySegment );
            };
            Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
         }
      }
      else {  // ColumnMajorOrder
         // There is a single kernel for the column-major organization,
         // so the launch configuration is only checked for an unsupported mapping.
         if constexpr( std::is_same_v< Device, Devices::GPU > )
            resolveLaunchConfiguration( launchConfig );
         const IndexType storageSize = segments.getStorageSize();
         const IndexType alignedSize = segments.getAlignedSize();
         auto l = [ = ] __cuda_callable__( const IndexType segmentIdx_idx ) mutable
         {
            TNL_ASSERT_LE( segmentIdx_idx, segmentIndexes_view.getSize(), "" );
            const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
            const IndexType begin = segmentIdx;
            const IndexType end = storageSize;
            ReturnType result = identity;
            IndexType argument = 0;
            IndexType localIdx = 0;
            for( IndexType j = begin; j < end; j += alignedSize, localIdx++ )
               reduction(
                  result,
                  detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, j ),
                  argument,
                  localIdx );
            bool emptySegment = ( segments.getSegmentSize() == 0 );
            storer( segmentIdx_idx, segmentIdx, argument, result, emptySegment );
         };
         Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
      }
   }
};

}  // namespace TNL::Algorithms::Segments::detail
