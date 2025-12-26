// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "FetchLambdaAdapter.h"
#include "ReducingKernels_Ellpack.h"
#include "ReducingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
struct ReducingOperations< EllpackView< Device, Index, Organization, Alignment > >
: public ReducingOperationsBase< EllpackView< Device, Index, Organization, Alignment > >
{
   using SegmentsViewType = EllpackView< Device, Index, Organization, Alignment >;
   using ConstViewType = typename SegmentsViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = typename std::remove_const< Index >::type;

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType,
             typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
   static void
   reduceSegments( const ConstViewType& segments,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
            if( end <= begin )
               return;
            const std::size_t threadsCount =
               (std::size_t) ( end - begin ) * (std::size_t) launchConfig.getThreadsPerSegmentCount();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );
            const Index blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            launch_config.gridSize.x = blocksCount;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
               constexpr auto kernel1 = EllpackCudaReductionKernel< 1,
                                                                    ConstViewType,
                                                                    IndexBegin,
                                                                    IndexEnd,
                                                                    std::remove_reference_t< Fetch >,
                                                                    std::remove_reference_t< Reduction >,
                                                                    std::remove_reference_t< ResultKeeper >,
                                                                    ReturnType >;
               constexpr auto kernel2 = EllpackCudaReductionKernel< 2,
                                                                    ConstViewType,
                                                                    IndexBegin,
                                                                    IndexEnd,
                                                                    std::remove_reference_t< Fetch >,
                                                                    std::remove_reference_t< Reduction >,
                                                                    std::remove_reference_t< ResultKeeper >,
                                                                    ReturnType >;
               constexpr auto kernel4 = EllpackCudaReductionKernel< 4,
                                                                    ConstViewType,
                                                                    IndexBegin,
                                                                    IndexEnd,
                                                                    std::remove_reference_t< Fetch >,
                                                                    std::remove_reference_t< Reduction >,
                                                                    std::remove_reference_t< ResultKeeper >,
                                                                    ReturnType >;
               constexpr auto kernel8 = EllpackCudaReductionKernel< 8,
                                                                    ConstViewType,
                                                                    IndexBegin,
                                                                    IndexEnd,
                                                                    std::remove_reference_t< Fetch >,
                                                                    std::remove_reference_t< Reduction >,
                                                                    std::remove_reference_t< ResultKeeper >,
                                                                    ReturnType >;
               constexpr auto kernel16 = EllpackCudaReductionKernel< 16,
                                                                     ConstViewType,
                                                                     IndexBegin,
                                                                     IndexEnd,
                                                                     std::remove_reference_t< Fetch >,
                                                                     std::remove_reference_t< Reduction >,
                                                                     std::remove_reference_t< ResultKeeper >,
                                                                     ReturnType >;
               constexpr auto kernel32 = EllpackCudaReductionKernel< 32,
                                                                     ConstViewType,
                                                                     IndexBegin,
                                                                     IndexEnd,
                                                                     std::remove_reference_t< Fetch >,
                                                                     std::remove_reference_t< Reduction >,
                                                                     std::remove_reference_t< ResultKeeper >,
                                                                     ReturnType >;

               switch( launchConfig.getThreadsPerSegmentCount() ) {
                  case 1:
                     Backend::launchKernelSync(
                        kernel1, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                     break;
                  case 2:
                     Backend::launchKernelSync(
                        kernel2, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                     break;
                  case 4:
                     Backend::launchKernelSync(
                        kernel4, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                     break;
                  case 8:
                     Backend::launchKernelSync(
                        kernel8, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                     break;
                  case 16:
                     Backend::launchKernelSync(
                        kernel16, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                     break;
                  case 32:
                     Backend::launchKernelSync(
                        kernel32, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
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
               keeper( segmentIdx, result );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {  // ColumnMajorOrder
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
            keeper( segmentIdx, aux );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }

   template< typename Array,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexes( const ConstViewType& segments,
                                     const Array& segmentIndexes,
                                     Fetch&& fetch,
                                     Reduction&& reduction,
                                     ResultKeeper&& keeper,
                                     const Value& identity,
                                     const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      using ArrayView = typename Array::ConstViewType;
      auto segmentIndexes_view = segmentIndexes.getConstView();
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
            if( segmentIndexes.getSize() == 0 )
               return;
            const std::size_t threadsCount =
               (std::size_t) segmentIndexes.getSize() * (std::size_t) launchConfig.getThreadsPerSegmentCount();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );
            const Index blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            launch_config.gridSize.x = blocksCount;
            constexpr auto kernel1 = EllpackCudaReductionKernelWithSegmentIndexes< 1,
                                                                                   ConstViewType,
                                                                                   ArrayView,
                                                                                   std::remove_reference_t< Fetch >,
                                                                                   std::remove_reference_t< Reduction >,
                                                                                   std::remove_reference_t< ResultKeeper >,
                                                                                   ReturnType >;
            constexpr auto kernel2 = EllpackCudaReductionKernelWithSegmentIndexes< 2,
                                                                                   ConstViewType,
                                                                                   ArrayView,
                                                                                   std::remove_reference_t< Fetch >,
                                                                                   std::remove_reference_t< Reduction >,
                                                                                   std::remove_reference_t< ResultKeeper >,
                                                                                   ReturnType >;
            constexpr auto kernel4 = EllpackCudaReductionKernelWithSegmentIndexes< 4,
                                                                                   ConstViewType,
                                                                                   ArrayView,
                                                                                   std::remove_reference_t< Fetch >,
                                                                                   std::remove_reference_t< Reduction >,
                                                                                   std::remove_reference_t< ResultKeeper >,
                                                                                   ReturnType >;
            constexpr auto kernel8 = EllpackCudaReductionKernelWithSegmentIndexes< 8,
                                                                                   ConstViewType,
                                                                                   ArrayView,
                                                                                   std::remove_reference_t< Fetch >,
                                                                                   std::remove_reference_t< Reduction >,
                                                                                   std::remove_reference_t< ResultKeeper >,
                                                                                   ReturnType >;
            constexpr auto kernel16 = EllpackCudaReductionKernelWithSegmentIndexes< 16,
                                                                                    ConstViewType,
                                                                                    ArrayView,
                                                                                    std::remove_reference_t< Fetch >,
                                                                                    std::remove_reference_t< Reduction >,
                                                                                    std::remove_reference_t< ResultKeeper >,
                                                                                    ReturnType >;
            constexpr auto kernel32 = EllpackCudaReductionKernelWithSegmentIndexes< 32,
                                                                                    ConstViewType,
                                                                                    ArrayView,
                                                                                    std::remove_reference_t< Fetch >,
                                                                                    std::remove_reference_t< Reduction >,
                                                                                    std::remove_reference_t< ResultKeeper >,
                                                                                    ReturnType >;
            switch( launchConfig.getThreadsPerSegmentCount() ) {
               case 1:
                  Backend::launchKernelSync(
                     kernel1, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 2:
                  Backend::launchKernelSync(
                     kernel2, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 4:
                  Backend::launchKernelSync(
                     kernel4, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 8:
                  Backend::launchKernelSync(
                     kernel8, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 16:
                  Backend::launchKernelSync(
                     kernel16, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 32:
                  Backend::launchKernelSync(
                     kernel32, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               default:
                  throw std::invalid_argument( "Unsupported threads per segment ( "
                                               + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                               + " ) count for Ellpack segments." );
                  break;
            }
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
               keeper( segmentIdx_idx, segmentIdx, result );
            };
            Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
         }
      }
      else {  // ColumnMajorOrder
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
            keeper( segmentIdx_idx, segmentIdx, result );
         };
         Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
      }
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument( const ConstViewType& segments,
                               IndexBegin begin,
                               IndexEnd end,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultKeeper&& keeper,
                               const Value& identity,
                               const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
            if( end <= begin )
               return;
            const std::size_t threadsCount =
               (std::size_t) ( end - begin ) * (std::size_t) launchConfig.getThreadsPerSegmentCount();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );
            const Index blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            launch_config.gridSize.x = blocksCount;
            constexpr auto kernel1 = EllpackCudaReductionKernelWithArgument< 1,
                                                                             ConstViewType,
                                                                             IndexBegin,
                                                                             IndexEnd,
                                                                             std::remove_reference_t< Fetch >,
                                                                             std::remove_reference_t< Reduction >,
                                                                             std::remove_reference_t< ResultKeeper >,
                                                                             ReturnType >;
            constexpr auto kernel2 = EllpackCudaReductionKernelWithArgument< 2,
                                                                             ConstViewType,
                                                                             IndexBegin,
                                                                             IndexEnd,
                                                                             std::remove_reference_t< Fetch >,
                                                                             std::remove_reference_t< Reduction >,
                                                                             std::remove_reference_t< ResultKeeper >,
                                                                             ReturnType >;
            constexpr auto kernel4 = EllpackCudaReductionKernelWithArgument< 4,
                                                                             ConstViewType,
                                                                             IndexBegin,
                                                                             IndexEnd,
                                                                             std::remove_reference_t< Fetch >,
                                                                             std::remove_reference_t< Reduction >,
                                                                             std::remove_reference_t< ResultKeeper >,
                                                                             ReturnType >;
            constexpr auto kernel8 = EllpackCudaReductionKernelWithArgument< 8,
                                                                             ConstViewType,
                                                                             IndexBegin,
                                                                             IndexEnd,
                                                                             std::remove_reference_t< Fetch >,
                                                                             std::remove_reference_t< Reduction >,
                                                                             std::remove_reference_t< ResultKeeper >,
                                                                             ReturnType >;
            constexpr auto kernel16 = EllpackCudaReductionKernelWithArgument< 16,
                                                                              ConstViewType,
                                                                              IndexBegin,
                                                                              IndexEnd,
                                                                              std::remove_reference_t< Fetch >,
                                                                              std::remove_reference_t< Reduction >,
                                                                              std::remove_reference_t< ResultKeeper >,
                                                                              ReturnType >;
            constexpr auto kernel32 = EllpackCudaReductionKernelWithArgument< 32,
                                                                              ConstViewType,
                                                                              IndexBegin,
                                                                              IndexEnd,
                                                                              std::remove_reference_t< Fetch >,
                                                                              std::remove_reference_t< Reduction >,
                                                                              std::remove_reference_t< ResultKeeper >,
                                                                              ReturnType >;
            switch( launchConfig.getThreadsPerSegmentCount() ) {
               case 1:
                  Backend::launchKernelSync( kernel1, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                  break;
               case 2:
                  Backend::launchKernelSync( kernel2, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                  break;
               case 4:
                  Backend::launchKernelSync( kernel4, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                  break;
               case 8:
                  Backend::launchKernelSync( kernel8, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                  break;
               case 16:
                  Backend::launchKernelSync(
                     kernel16, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                  break;
               case 32:
                  Backend::launchKernelSync(
                     kernel32, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
                  break;
               default:
                  throw std::invalid_argument( "Unsupported threads per segment ( "
                                               + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                               + " ) count for Ellpack segments." );
                  break;
            }
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
                  reduction( result,
                             detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, j ),
                             argument,
                             localIdx );
               keeper( segmentIdx, argument, result );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {  // ColumnMajorOrder
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
               reduction( result,
                          detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, j ),
                          argument,
                          localIdx );
            keeper( segmentIdx, argument, result );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }

   template< typename Array,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexesAndArgument( const ConstViewType& segments,
                                                const Array& segmentIndexes,
                                                Fetch&& fetch,
                                                Reduction&& reduction,
                                                ResultKeeper&& keeper,
                                                const Value& identity,
                                                const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      using ArrayView = typename Array::ConstViewType;
      auto segmentIndexes_view = segmentIndexes.getConstView();
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
            if( segmentIndexes.getSize() == 0 )
               return;
            const std::size_t threadsCount =
               (std::size_t) segmentIndexes.getSize() * (std::size_t) launchConfig.getThreadsPerSegmentCount();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );
            const std::size_t blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            launch_config.gridSize.x = blocksCount;
            constexpr auto kernel1 =
               EllpackCudaReductionKernelWithSegmentIndexesAndArgument< 1,
                                                                        ConstViewType,
                                                                        ArrayView,
                                                                        std::remove_reference_t< Fetch >,
                                                                        std::remove_reference_t< Reduction >,
                                                                        std::remove_reference_t< ResultKeeper >,
                                                                        ReturnType >;
            constexpr auto kernel2 =
               EllpackCudaReductionKernelWithSegmentIndexesAndArgument< 2,
                                                                        ConstViewType,
                                                                        ArrayView,
                                                                        std::remove_reference_t< Fetch >,
                                                                        std::remove_reference_t< Reduction >,
                                                                        std::remove_reference_t< ResultKeeper >,
                                                                        ReturnType >;
            constexpr auto kernel4 =
               EllpackCudaReductionKernelWithSegmentIndexesAndArgument< 4,
                                                                        ConstViewType,
                                                                        ArrayView,
                                                                        std::remove_reference_t< Fetch >,
                                                                        std::remove_reference_t< Reduction >,
                                                                        std::remove_reference_t< ResultKeeper >,
                                                                        ReturnType >;
            constexpr auto kernel8 =
               EllpackCudaReductionKernelWithSegmentIndexesAndArgument< 8,
                                                                        ConstViewType,
                                                                        ArrayView,
                                                                        std::remove_reference_t< Fetch >,
                                                                        std::remove_reference_t< Reduction >,
                                                                        std::remove_reference_t< ResultKeeper >,
                                                                        ReturnType >;
            constexpr auto kernel16 =
               EllpackCudaReductionKernelWithSegmentIndexesAndArgument< 16,
                                                                        ConstViewType,
                                                                        ArrayView,
                                                                        std::remove_reference_t< Fetch >,
                                                                        std::remove_reference_t< Reduction >,
                                                                        std::remove_reference_t< ResultKeeper >,
                                                                        ReturnType >;
            constexpr auto kernel32 =
               EllpackCudaReductionKernelWithSegmentIndexesAndArgument< 32,
                                                                        ConstViewType,
                                                                        ArrayView,
                                                                        std::remove_reference_t< Fetch >,
                                                                        std::remove_reference_t< Reduction >,
                                                                        std::remove_reference_t< ResultKeeper >,
                                                                        ReturnType >;
            switch( launchConfig.getThreadsPerSegmentCount() ) {
               case 1:
                  Backend::launchKernelSync(
                     kernel1, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 2:
                  Backend::launchKernelSync(
                     kernel2, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 4:
                  Backend::launchKernelSync(
                     kernel4, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 8:
                  Backend::launchKernelSync(
                     kernel8, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 16:
                  Backend::launchKernelSync(
                     kernel16, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               case 32:
                  Backend::launchKernelSync(
                     kernel32, launch_config, segments, segmentIndexes.getConstView(), fetch, reduction, keeper, identity );
                  break;
               default:
                  throw std::invalid_argument( "Unsupported threads per segment ( "
                                               + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                               + " ) count for Ellpack segments." );
                  break;
            }
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
                  reduction( result,
                             detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, j ),
                             argument,
                             localIdx );
               keeper( segmentIdx_idx, segmentIdx, argument, result );
            };
            Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
         }
      }
      else {  // ColumnMajorOrder
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
               reduction( result,
                          detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, j ),
                          argument,
                          localIdx );
            keeper( segmentIdx_idx, segmentIdx, argument, result );
         };
         Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
      }
   }
};

}  //namespace TNL::Algorithms::Segments::detail
