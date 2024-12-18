// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/Ellpack.h>

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
struct SegmentsOperations< EllpackView< Device, Index, Organization, Alignment > >
{
   using ViewType = EllpackView< Device, Index, Organization, Alignment >;
   // ViewType is the same as ConstViewType for Ellpack !!!!!
   //using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = Index;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      if constexpr( Organization == RowMajorOrder ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
         const IndexType segmentSize = segments.getSegmentSize();
         if constexpr( argumentCount< Function >() == 3 ) {
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, localIdx++, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {
         const IndexType storageSize = segments.getStorageSize();
         const IndexType alignedSize = segments.getAlignedSize();
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType begin = segmentIdx;
               const IndexType end = storageSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, localIdx++, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               const IndexType begin = segmentIdx;
               const IndexType end = storageSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      auto segmentIndexesView = segmentIndexes.getConstView();
      if constexpr( Organization == RowMajorOrder ) {
         const IndexType segmentSize = segments.getSegmentSize();
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType idx ) mutable
            {
               const IndexType segmentIdx = segmentIndexesView[ idx ];
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, localIdx++, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType idx ) mutable
            {
               const IndexType segmentIdx = segmentIndexesView[ idx ];
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {
         const IndexType storageSize = segments.getStorageSize();
         const IndexType alignedSize = segments.getAlignedSize();
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType idx ) mutable
            {
               const IndexType segmentIdx = segmentIndexesView[ idx ];
               const IndexType begin = segmentIdx;
               const IndexType end = storageSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, localIdx++, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType idx ) mutable
            {
               const IndexType segmentIdx = segmentIndexesView[ idx ];
               const IndexType begin = segmentIdx;
               const IndexType end = storageSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ViewType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  const LaunchConfiguration& launchConfig,
                  Condition condition,
                  Function function )
   {
      if constexpr( Organization == RowMajorOrder ) {
         const IndexType segmentSize = segments.getSegmentSize();
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               if( ! condition( segmentIdx ) )
                  return;
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, localIdx++, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               if( ! condition( segmentIdx ) )
                  return;
               const IndexType begin = segmentIdx * segmentSize;
               const IndexType end = begin + segmentSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
      else {
         const IndexType storageSize = segments.getStorageSize();
         const IndexType alignedSize = segments.getAlignedSize();
         if constexpr( argumentCount< Function >() == 3 ) {  // TODO: Move this inside the lambda function when nvcc accepts it.
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               if( ! condition( segmentIdx ) )
                  return;
               const IndexType begin = segmentIdx;
               const IndexType end = storageSize;
               IndexType localIdx( 0 );
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, localIdx++, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
         else {  // argumentCount< Function >() == 2
            auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
            {
               if( ! condition( segmentIdx ) )
                  return;
               const IndexType begin = segmentIdx;
               const IndexType end = storageSize;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
                  function( segmentIdx, globalIdx );
            };
            Algorithms::parallelFor< Device >( begin, end, l );
         }
      }
   }
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
struct SegmentsOperations< Ellpack< Device, Index, IndexAllocator, Organization, Alignment > >
{
   using SegmentsType = Ellpack< Device, Index, IndexAllocator, Organization, Alignment >;
   using ViewType = typename SegmentsType::ViewType;
   using ConstViewType = typename SegmentsType::ViewType;
   using DeviceType = Device;
   using IndexType = Index;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const SegmentsType& segments,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      SegmentsOperations< ViewType >::forElements(
         segments.getConstView(), begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const SegmentsType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                const LaunchConfiguration& launchConfig,
                Function&& function )
   {
      SegmentsOperations< ViewType >::forElements(
         segments.getConstView(), segmentIndexes, begin, end, launchConfig, std::forward< Function >( function ) );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const SegmentsType& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  const LaunchConfiguration& launchConfig,
                  Condition&& condition,
                  Function&& function )
   {
      SegmentsOperations< ViewType >::forElementsIf( segments.getConstView(),
                                                     begin,
                                                     end,
                                                     launchConfig,
                                                     std::forward< Condition >( condition ),
                                                     std::forward< Function >( function ) );
   }
};
}  //namespace TNL::Algorithms::Segments::detail
