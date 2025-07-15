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
                   Fetch fetch,          // TODO Fetch&& does not work with nvcc
                   Reduction reduction,  // TODO Reduction&& does not work with nvcc
                   ResultKeeper keeper,  // TODO ResultKeeper&& does not work with nvcc
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
            if( end <= begin )
               return;
            const Index segmentsCount = end - begin;
            const Index threadsCount = segmentsCount * Backend::getWarpSize();
            const Index blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            launch_config.gridSize.x = blocksCount;
            constexpr auto kernel =
               EllpackCudaReductionKernel< ConstViewType, IndexBegin, IndexEnd, Fetch, Reduction, ResultKeeper, ReturnType >;
            Backend::launchKernelSync( kernel, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
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
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexes( const ConstViewType& segments,
                                     const Array& segmentIndexes,
                                     IndexBegin begin,
                                     IndexEnd end,
                                     Fetch fetch,          // TODO Fetch&& does not work with nvcc
                                     Reduction reduction,  // TODO Reduction&& does not work with nvcc
                                     ResultKeeper keeper,  // TODO ResultKeeper&& does not work with nvcc
                                     const Value& identity,
                                     const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      using ArrayView = typename Array::ConstViewType;
      auto segmentIndexes_view = segmentIndexes.getConstView();
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
            if( end <= begin )
               return;
            const Index segmentsCount = end - begin;
            const Index threadsCount = segmentsCount * Backend::getWarpSize();
            const Index blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            launch_config.gridSize.x = blocksCount;
            constexpr auto kernel = EllpackCudaReductionKernelWithSegmentIndexes< ConstViewType,
                                                                                  ArrayView,
                                                                                  IndexBegin,
                                                                                  IndexEnd,
                                                                                  Fetch,
                                                                                  Reduction,
                                                                                  ResultKeeper,
                                                                                  ReturnType >;
            Backend::launchKernelSync(
               kernel, launch_config, segments, segmentIndexes.getConstView(), begin, end, fetch, reduction, keeper, identity );
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
            Algorithms::parallelFor< Device >( begin, end, l );
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
         Algorithms::parallelFor< Device >( begin, end, l );
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
                               Fetch fetch,          // TODO Fetch&& does not work with nvcc
                               Reduction reduction,  // TODO Reduction&& does not work with nvcc
                               ResultKeeper keeper,  // TODO ResultKeeper&& does not work with nvcc
                               const Value& identity,
                               const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
            if( end <= begin )
               return;
            const Index segmentsCount = end - begin;
            const Index threadsCount = segmentsCount * Backend::getWarpSize();
            const Index blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            launch_config.gridSize.x = blocksCount;
            constexpr auto kernel = EllpackCudaReductionKernelWithArgument< ConstViewType,
                                                                            IndexBegin,
                                                                            IndexEnd,
                                                                            Fetch,
                                                                            Reduction,
                                                                            ResultKeeper,
                                                                            ReturnType >;
            Backend::launchKernelSync( kernel, launch_config, segments, begin, end, fetch, reduction, keeper, identity );
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
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithSegmentIndexesAndArgument( const ConstViewType& segments,
                                                const Array& segmentIndexes,
                                                IndexBegin begin,
                                                IndexEnd end,
                                                Fetch fetch,          // TODO Fetch&& does not work with nvcc
                                                Reduction reduction,  // TODO Reduction&& does not work with nvcc
                                                ResultKeeper keeper,  // TODO ResultKeeper&& does not work with nvcc
                                                const Value& identity,
                                                const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      using ArrayView = typename Array::ConstViewType;
      auto segmentIndexes_view = segmentIndexes.getConstView();
      if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
         if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
            if( end <= begin )
               return;
            const Index segmentsCount = end - begin;
            const Index threadsCount = segmentsCount * Backend::getWarpSize();
            const Index blocksCount = Backend::getNumberOfBlocks( threadsCount, 256 );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            launch_config.gridSize.x = blocksCount;
            constexpr auto kernel = EllpackCudaReductionKernelWithSegmentIndexesAndArgument< ConstViewType,
                                                                                             ArrayView,
                                                                                             IndexBegin,
                                                                                             IndexEnd,
                                                                                             Fetch,
                                                                                             Reduction,
                                                                                             ResultKeeper,
                                                                                             ReturnType >;
            Backend::launchKernelSync(
               kernel, launch_config, segments, segmentIndexes.getConstView(), begin, end, fetch, reduction, keeper, identity );
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
            Algorithms::parallelFor< Device >( begin, end, l );
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
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }
};

}  //namespace TNL::Algorithms::Segments::detail
