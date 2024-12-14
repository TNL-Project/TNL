// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/BiEllpackView.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
struct SegmentsOperations< BiEllpackView< Device, Index, Organization, WarpSize > >
{
   using ViewType = BiEllpackView< Device, Index, Organization, WarpSize >;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = Index;
   using ConstOffsetsView = typename ViewType::ConstOffsetsView;

   [[nodiscard]] static constexpr int
   getWarpSize()
   {
      return WarpSize;
   }

   [[nodiscard]] static constexpr int
   getLogWarpSize()
   {
      return TNL::discreteLog2( getWarpSize() );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments, IndexBegin begin, IndexEnd end, Function&& function )
   {
      const auto segmentsPermutationView = segments.getSegmentsPermutationView();
      const auto groupPointersView = segments.getGroupPointersView();
      auto work = [ segmentsPermutationView, groupPointersView, function ] __cuda_callable__( IndexType segmentIdx ) mutable
      {
         const IndexType strip = segmentIdx / getWarpSize();
         const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
         const IndexType segmentStripPerm = segmentsPermutationView[ segmentIdx ] - strip * getWarpSize();
         const IndexType groupsCount =
            detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCountDirect(
               segmentsPermutationView, segmentIdx );
         IndexType groupHeight = getWarpSize();
         IndexType localIdx = 0;
         for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
            IndexType groupOffset = groupPointersView[ groupIdx ];
            const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
            if( groupSize ) {
               const IndexType groupWidth = groupSize / groupHeight;
               for( IndexType i = 0; i < groupWidth; i++ ) {
                  if constexpr( argumentCount< Function >() == 3 ) {
                     if constexpr( Organization == RowMajorOrder ) {
                        function( segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i );
                     }
                     else {
                        function( segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight );
                     }
                     localIdx++;
                  }
                  else  // argumentCount< Function >() == 2
                     if constexpr( Organization == RowMajorOrder ) {
                        function( segmentIdx, groupOffset + segmentStripPerm * groupWidth + i );
                     }
                     else {
                        function( segmentIdx, groupOffset + segmentStripPerm + i * groupHeight );
                     }
               }
            }
            groupHeight /= 2;
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, work );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments, IndexBegin begin, IndexEnd end, Function&& function )
   {
      return forElements( segments.getConstView(), begin, end, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ConstViewType& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                Function&& function )
   {
      const auto segmentsPermutationView = segments.getSegmentsPermutationView();
      const auto groupPointersView = segments.getGroupPointersView();
      auto segmentIndexesView = segmentIndexes.getConstView();
      auto work =
         [ segmentIndexesView, segmentsPermutationView, groupPointersView, function ] __cuda_callable__( IndexType idx ) mutable
      {
         const IndexType segmentIdx = segmentIndexesView[ idx ];
         const IndexType strip = segmentIdx / getWarpSize();
         const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
         const IndexType segmentStripPerm = segmentsPermutationView[ segmentIdx ] - strip * getWarpSize();
         const IndexType groupsCount =
            detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCountDirect(
               segmentsPermutationView, segmentIdx );
         IndexType groupHeight = getWarpSize();
         IndexType localIdx = 0;
         for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
            IndexType groupOffset = groupPointersView[ groupIdx ];
            const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
            if( groupSize ) {
               const IndexType groupWidth = groupSize / groupHeight;
               for( IndexType i = 0; i < groupWidth; i++ ) {
                  if constexpr( argumentCount< Function >() == 3 ) {
                     if constexpr( Organization == RowMajorOrder ) {
                        function( segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i );
                     }
                     else {
                        function( segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight );
                     }
                     localIdx++;
                  }
                  else  // argumentCount< Function >() == 2
                     if constexpr( Organization == RowMajorOrder ) {
                        function( segmentIdx, groupOffset + segmentStripPerm * groupWidth + i );
                     }
                     else {
                        function( segmentIdx, groupOffset + segmentStripPerm + i * groupHeight );
                     }
               }
            }
            groupHeight /= 2;
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, work );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const ViewType& segments, const Array& segmentIndexes, IndexBegin begin, IndexEnd end, Function&& function )
   {
      return forElements( segments.getConstView(), segmentIndexes, begin, end, std::forward< Function >( function ) );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ConstViewType& segments, IndexBegin begin, IndexEnd end, Condition condition, Function function )
   {
      const auto segmentsPermutationView = segments.getSegmentsPermutationView();
      const auto groupPointersView = segments.getGroupPointersView();
      auto work =
         [ segmentsPermutationView, groupPointersView, condition, function ] __cuda_callable__( IndexType segmentIdx ) mutable
      {
         if( ! condition( segmentIdx ) )
            return;
         const IndexType strip = segmentIdx / getWarpSize();
         const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
         const IndexType segmentStripPerm = segmentsPermutationView[ segmentIdx ] - strip * getWarpSize();
         const IndexType groupsCount =
            detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCountDirect(
               segmentsPermutationView, segmentIdx );
         IndexType groupHeight = getWarpSize();
         IndexType localIdx = 0;
         for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ ) {
            IndexType groupOffset = groupPointersView[ groupIdx ];
            const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
            if( groupSize ) {
               const IndexType groupWidth = groupSize / groupHeight;
               for( IndexType i = 0; i < groupWidth; i++ ) {
                  if constexpr( Organization == RowMajorOrder ) {
                     function( segmentIdx, localIdx, groupOffset + segmentStripPerm * groupWidth + i );
                  }
                  else {
                     function( segmentIdx, localIdx, groupOffset + segmentStripPerm + i * groupHeight );
                  }
                  localIdx++;
               }
            }
            groupHeight /= 2;
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, work );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const ViewType& segments, IndexBegin begin, IndexEnd end, Condition condition, Function function )
   {
      forElementsIf(
         segments.getConstView(), begin, end, std::forward< Condition >( condition ), std::forward< Function >( function ) );
   }
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
struct SegmentsOperations< Segments::BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize > >
{
   // TODO: Rename BiEllpack in the detail folder
   using SegmentsType = Segments::BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >;
   using ViewType = typename SegmentsType::ViewType;
   using ConstViewType = typename SegmentsType::ViewType;
   using DeviceType = Device;
   using IndexType = Index;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const SegmentsType& segments, IndexBegin begin, IndexEnd end, Function&& function )
   {
      SegmentsOperations< ViewType >::forElements( segments.getConstView(), begin, end, std::forward< Function >( function ) );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forElements( const SegmentsType& segments, const Array& segmentIndexes, IndexBegin begin, IndexEnd end, Function&& function )
   {
      SegmentsOperations< ViewType >::forElements(
         segments.getConstView(), segmentIndexes, begin, end, std::forward< Function >( function ) );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf( const SegmentsType& segments, IndexBegin begin, IndexEnd end, Condition&& condition, Function&& function )
   {
      SegmentsOperations< ViewType >::forElementsIf(
         segments.getConstView(), begin, end, std::forward< Condition >( condition ), std::forward< Function >( function ) );
   }
};
}  //namespace TNL::Algorithms::Segments::detail