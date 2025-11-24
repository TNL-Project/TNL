// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include "FetchLambdaAdapter.h"
#include "ReducingOperationsBase.h"
#include "ReducingKernels_SlicedEllpack.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Segments, int ThreadsPerSegment, int BlockSize >
static constexpr bool
SlicedEllpackReductionSupported()
{
   return Segments::getOrganization() == ElementsOrganization::RowMajorOrder
       || ( Segments::getSliceSize() * ThreadsPerSegment <= 256
            && Segments::getSliceSize() * ThreadsPerSegment >= Backend::getWarpSize() );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
struct ReducingOperations< SlicedEllpackView< Device, Index, Organization, SliceSize > >
: public ReducingOperationsBase< SlicedEllpackView< Device, Index, Organization, SliceSize > >
{
   using SegmentsViewType = SlicedEllpackView< Device, Index, Organization, SliceSize >;
   using ConstViewType = typename SegmentsViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = typename std::remove_const< Index >::type;
   using ConstOffsetsView = typename SegmentsViewType::ConstOffsetsView;

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsSequential( const ConstViewType& segments,
                             IndexBegin begin,
                             IndexEnd end,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             ResultKeeper&& keeper,
                             const Value& identity,
                             const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

      const auto sliceSegmentSizes = segments.getSliceSegmentSizesView();
      const auto sliceOffsets = segments.getSliceOffsetsView();

      auto l = [ sliceOffsets, sliceSegmentSizes, fetch, reduction, keeper, identity ] __cuda_callable__(
                  const IndexType segmentIdx ) mutable
      {
         const IndexType sliceIdx = segmentIdx / SegmentsViewType::getSliceSize();
         const IndexType segmentInSliceIdx = segmentIdx % SegmentsViewType::getSliceSize();
         ReturnType aux = identity;
         IndexType localIdx = 0;

         if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes[ sliceIdx ];
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               aux = reduction(
                  aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
         }
         else {
            (void) sliceSegmentSizes;  // ignore warning due to unused capture - let the compiler optimize it out...
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets[ sliceIdx + 1 ];

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SegmentsViewType::getSliceSize() )
               aux = reduction(
                  aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
         }
         keeper( segmentIdx, aux );
      };

      Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegments( const ConstViewType& segments,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch fetch,          // TODO: && does not work here for some reason
                   Reduction reduction,  // TODO: && does not work here for some reason
                   ResultKeeper keeper,  // TODO: && does not work here for some reason
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      if constexpr( std::is_same_v< Device, TNL::Devices::Host > || std::is_same_v< Device, TNL::Devices::Sequential >
                    || is_complex_v< Value > )  // Complex numbers are not supported in CUDA kernels due to use of shfl.
      {
         reduceSegmentsSequential( segments,
                                   begin,
                                   end,
                                   std::forward< Fetch >( fetch ),
                                   std::forward< Reduction >( reduction ),
                                   std::forward< ResultKeeper >( keeper ),
                                   identity,
                                   launchConfig );
      }
      else {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
             && launchConfig.getThreadsPerSegmentCount() == 1 )
         {
            reduceSegmentsSequential( segments,
                                      begin,
                                      end,
                                      std::forward< Fetch >( fetch ),
                                      std::forward< Reduction >( reduction ),
                                      std::forward< ResultKeeper >( keeper ),
                                      identity,
                                      launchConfig );
            return;
         }
         std::size_t sliceCount = end / SliceSize + ( end % SliceSize != 0 ) - begin / SliceSize;
         TNL_ASSERT_LE( sliceCount, (std::size_t) segments.getSliceSegmentSizesView().getSize(), "Too many slices." );
         std::size_t threadsCount = sliceCount * ConstViewType::getSliceSize();
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Warp )
            threadsCount *= (std::size_t) Backend::getWarpSize();
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed )
            threadsCount *= (std::size_t) launchConfig.getThreadsPerSegmentCount();
         if( threadsCount > std::numeric_limits< IndexType >::max() )
            throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );

         Backend::LaunchConfiguration launch_config;
         launch_config.blockSize.x = 256;
         dim3 blocksCount;
         dim3 gridsCount;
         Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
         for( IndexType gridIdx = 0; gridIdx < (Index) gridsCount.x; gridIdx++ ) {
            Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
               switch( launchConfig.getThreadsPerSegmentCount() ) {
                  case 2:
                     {
                        if constexpr( SlicedEllpackReductionSupported< ConstViewType, 2, 256 >() )

                        {
                           constexpr auto kernel = reduceSegmentsSlicedEllpackKernel< 256,
                                                                                      2,
                                                                                      ConstViewType,
                                                                                      IndexType,
                                                                                      IndexType,
                                                                                      Fetch,
                                                                                      Reduction,
                                                                                      ResultKeeper,
                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                        }
                        else
                           throw std::runtime_error(
                              "Wrong configuration of GPU threads for reduction in SlicedEllpak: organization = "
                              + std::string( SegmentsViewType::getOrganization() == Segments::RowMajorOrder
                                                ? "RowMajorOrder"
                                                : "ColumnMajorOrder" )
                              + ", SliceSize = " + std::to_string( SegmentsViewType::getSliceSize() )
                              + " TPS = 2, warp size = " + std::to_string( Backend::getWarpSize() ) + "." );
                        break;
                     }
                  case 4:
                     {
                        if constexpr( SlicedEllpackReductionSupported< ConstViewType, 4, 256 >() )

                        {
                           constexpr auto kernel = reduceSegmentsSlicedEllpackKernel< 256,
                                                                                      4,
                                                                                      ConstViewType,
                                                                                      IndexType,
                                                                                      IndexType,
                                                                                      Fetch,
                                                                                      Reduction,
                                                                                      ResultKeeper,
                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                        }
                        else
                           throw std::runtime_error(
                              "Wrong configuration of GPU threads for reduction in SlicedEllpak: organization = "
                              + std::string( SegmentsViewType::getOrganization() == Segments::RowMajorOrder
                                                ? "RowMajorOrder"
                                                : "ColumnMajorOrder" )
                              + ", SliceSize = " + std::to_string( SegmentsViewType::getSliceSize() )
                              + " TPS = 4, warp size = " + std::to_string( Backend::getWarpSize() ) + "." );

                        break;
                     }
                  case 8:
                     {
                        if constexpr( SlicedEllpackReductionSupported< ConstViewType, 8, 256 >() )

                        {
                           constexpr auto kernel = reduceSegmentsSlicedEllpackKernel< 256,
                                                                                      8,
                                                                                      ConstViewType,
                                                                                      IndexType,
                                                                                      IndexType,
                                                                                      Fetch,
                                                                                      Reduction,
                                                                                      ResultKeeper,
                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                        }
                        else
                           throw std::runtime_error(
                              "Wrong configuration of GPU threads for reduction in SlicedEllpak: organization = "
                              + std::string( SegmentsViewType::getOrganization() == Segments::RowMajorOrder
                                                ? "RowMajorOrder"
                                                : "ColumnMajorOrder" )
                              + ", SliceSize = " + std::to_string( SegmentsViewType::getSliceSize() )
                              + " TPS = 8, warp size = " + std::to_string( Backend::getWarpSize() ) + "." );

                        break;
                     }
                  case 16:
                     {
                        if constexpr( SlicedEllpackReductionSupported< ConstViewType, 16, 256 >() )

                        {
                           constexpr auto kernel = reduceSegmentsSlicedEllpackKernel< 256,
                                                                                      16,
                                                                                      ConstViewType,
                                                                                      IndexType,
                                                                                      IndexType,
                                                                                      Fetch,
                                                                                      Reduction,
                                                                                      ResultKeeper,
                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                        }
                        else
                           throw std::runtime_error(
                              "Wrong configuration of GPU threads for reduction in SlicedEllpak: organization = "
                              + std::string( SegmentsViewType::getOrganization() == Segments::RowMajorOrder
                                                ? "RowMajorOrder"
                                                : "ColumnMajorOrder" )
                              + ", SliceSize = " + std::to_string( SegmentsViewType::getSliceSize() )
                              + " TPS = 16, warp size = " + std::to_string( Backend::getWarpSize() ) + "." );

                        break;
                     }
                  case 32:
                     {
                        if constexpr( SlicedEllpackReductionSupported< ConstViewType, 32, 256 >() )

                        {
                           constexpr auto kernel = reduceSegmentsSlicedEllpackKernel< 256,
                                                                                      32,
                                                                                      ConstViewType,
                                                                                      IndexType,
                                                                                      IndexType,
                                                                                      Fetch,
                                                                                      Reduction,
                                                                                      ResultKeeper,
                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                        }
                        else
                           throw std::runtime_error(
                              "Wrong configuration of GPU threads for reduction in SlicedEllpak: organization = "
                              + std::string( SegmentsViewType::getOrganization() == Segments::RowMajorOrder
                                                ? "RowMajorOrder"
                                                : "ColumnMajorOrder" )
                              + ", SliceSize = " + std::to_string( SegmentsViewType::getSliceSize() )
                              + " TPS = 32, warp size = " + std::to_string( Backend::getWarpSize() ) + "." );

                        break;
                     }
                  default:
                     throw std::runtime_error( "Unsupported number of threads per segment"
                                               + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                               + ". It can be only 2, 4, 8, 16 or 32." );
               }
            }
            else {
               throw std::runtime_error( "Unsupported threads to segments mapping strategy." );
            }
         }
         Backend::streamSynchronize( launch_config.stream );
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
                                     Fetch&& fetch,
                                     Reduction&& reduction,
                                     ResultKeeper&& keeper,
                                     const Value& identity,
                                     const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

      const auto sliceSegmentSizes = segments.getSliceSegmentSizesView();
      const auto sliceOffsets = segments.getSliceOffsetsView();
      auto segmentIndexes_view = segmentIndexes.getConstView();

      auto l = [ sliceOffsets, segmentIndexes_view, sliceSegmentSizes, fetch, reduction, keeper, identity ] __cuda_callable__(
                  const IndexType segmentIdx_idx ) mutable
      {
         TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes_view.getSize(), "" );
         const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
         const IndexType sliceIdx = segmentIdx / SegmentsViewType::getSliceSize();
         const IndexType segmentInSliceIdx = segmentIdx % SegmentsViewType::getSliceSize();
         ReturnType result = identity;
         IndexType localIdx = 0;

         if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes[ sliceIdx ];
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               result = reduction(
                  result, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
         }
         else {
            (void) sliceSegmentSizes;  // ignore warning due to unused capture - let the compiler
                                       // optimize it out...
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets[ sliceIdx + 1 ];

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SegmentsViewType::getSliceSize() )
               result = reduction(
                  result, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx ) );
         }
         keeper( segmentIdx_idx, segmentIdx, result );
      };

      Algorithms::parallelFor< Device >( begin, end, l );
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

      const auto sliceSegmentSizes = segments.getSliceSegmentSizesView();
      const auto sliceOffsets = segments.getSliceOffsetsView();

      auto l = [ sliceOffsets, sliceSegmentSizes, fetch, reduction, keeper, identity ] __cuda_callable__(
                  const IndexType segmentIdx ) mutable
      {
         const IndexType sliceIdx = segmentIdx / SegmentsViewType::getSliceSize();
         const IndexType segmentInSliceIdx = segmentIdx % SegmentsViewType::getSliceSize();
         ReturnType result = identity;
         IndexType argument = 0;
         IndexType localIdx = 0;

         if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes[ sliceIdx ];
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++, localIdx++ )
               reduction( result,
                          detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                          argument,
                          localIdx );
         }
         else {
            (void) sliceSegmentSizes;  // ignore warning due to unused capture - let the compiler
                                       // optimize it out...
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets[ sliceIdx + 1 ];

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SegmentsViewType::getSliceSize(), localIdx++ )
               reduction( result,
                          detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                          argument,
                          localIdx );
         }
         keeper( segmentIdx, argument, result );
      };

      Algorithms::parallelFor< Device >( begin, end, l );
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
                                                Fetch&& fetch,
                                                Reduction&& reduction,
                                                ResultKeeper&& keeper,
                                                const Value& identity,
                                                const LaunchConfiguration& launchConfig )
   {
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

      const auto sliceSegmentSizes = segments.getSliceSegmentSizesView();
      const auto sliceOffsets = segments.getSliceOffsetsView();
      auto segmentIndexes_view = segmentIndexes.getConstView();

      auto l = [ sliceOffsets, segmentIndexes_view, sliceSegmentSizes, fetch, reduction, keeper, identity ] __cuda_callable__(
                  const IndexType segmentIdx_idx ) mutable
      {
         TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes_view.getSize(), "" );
         const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
         const IndexType sliceIdx = segmentIdx / SegmentsViewType::getSliceSize();
         const IndexType segmentInSliceIdx = segmentIdx % SegmentsViewType::getSliceSize();
         ReturnType result = identity;
         IndexType argument = 0;
         IndexType localIdx = 0;

         if constexpr( SegmentsViewType::getOrganization() == Segments::RowMajorOrder ) {
            const IndexType segmentSize = sliceSegmentSizes[ sliceIdx ];
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx * segmentSize;
            const IndexType end = begin + segmentSize;

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++, localIdx++ )
               reduction( result,
                          detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                          argument,
                          localIdx );
         }
         else {
            (void) sliceSegmentSizes;  // ignore warning due to unused capture - let the compiler
                                       // optimize it out...
            const IndexType begin = sliceOffsets[ sliceIdx ] + segmentInSliceIdx;
            const IndexType end = sliceOffsets[ sliceIdx + 1 ];

            for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SegmentsViewType::getSliceSize(), localIdx++ )
               reduction( result,
                          detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx ),
                          argument,
                          localIdx );
         }
         keeper( segmentIdx_idx, segmentIdx, argument, result );
      };

      Algorithms::parallelFor< Device >( begin, end, l );
   }
};

}  //namespace TNL::Algorithms::Segments::detail
