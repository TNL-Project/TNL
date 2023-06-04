// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Segments/ChunkedEllpackSegmentView.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * In the ChunkedEllpack, the segments are split into slices. This is done
 * in ChunkedEllpack::resolveSliceSizes. All segments elements in each slice
 * are split into chunks. All chunks in one slice have the same size, but the size
 * of chunks can be different in each slice.
 */
template< typename Index >
struct ChunkedEllpackSliceInfo
{
   //! \brief The size of the slice, it means the number of the segments
   //! covered by the slice.
   Index size;

   //! \brief The chunk size, i.e. maximal number of non-zero elements that can
   //! be stored in the chunk.
   Index chunkSize;

   //! \brief Index of the first segment covered be this slice.
   Index firstSegment;

   //! \brief Position of the first element of this slice.
   Index pointer;
};

template< typename Index, typename Device, ElementsOrganization Organization >
class ChunkedEllpack
{
public:
   using DeviceType = Device;
   using IndexType = Index;
   using ConstOffsetsView = Containers::VectorView< std::add_const_t< Index >, DeviceType, IndexType >;
   using SliceInfoType = ChunkedEllpackSliceInfo< IndexType >;
   using ConstSliceInfoContainerView = Containers::ArrayView< std::add_const_t< SliceInfoType >, DeviceType, IndexType >;
   using SegmentViewType = ChunkedEllpackSegmentView< IndexType, Organization >;

   [[nodiscard]] __cuda_callable__
   static IndexType
   getSegmentSizeDirect( const ConstOffsetsView& segmentsToSlicesMapping,
                         const ConstSliceInfoContainerView& slices,
                         const ConstOffsetsView& segmentsToChunksMapping,
                         const IndexType segmentIdx )
   {
      const IndexType& sliceIndex = segmentsToSlicesMapping[ segmentIdx ];
      IndexType firstChunkOfSegment = 0;
      if( segmentIdx != slices[ sliceIndex ].firstSegment )
         firstChunkOfSegment = segmentsToChunksMapping[ segmentIdx - 1 ];

      const IndexType lastChunkOfSegment = segmentsToChunksMapping[ segmentIdx ];
      const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
      const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
      return chunkSize * segmentChunksCount;
   }

   [[nodiscard]] static IndexType
   getSegmentSize( const ConstOffsetsView& segmentsToSlicesMapping,
                   const ConstSliceInfoContainerView& slices,
                   const ConstOffsetsView& segmentsToChunksMapping,
                   const IndexType segmentIdx )
   {
      const IndexType& sliceIndex = segmentsToSlicesMapping.getElement( segmentIdx );
      IndexType firstChunkOfSegment = 0;
      if( segmentIdx != slices.getElement( sliceIndex ).firstSegment )
         firstChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx - 1 );

      const IndexType lastChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx );
      const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
      const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
      return chunkSize * segmentChunksCount;
   }

   [[nodiscard]] __cuda_callable__
   static IndexType
   getGlobalIndexDirect( const ConstOffsetsView& segmentsToSlicesMapping,
                         const ConstSliceInfoContainerView& slices,
                         const ConstOffsetsView& segmentsToChunksMapping,
                         const IndexType chunksInSlice,
                         const IndexType segmentIdx,
                         const IndexType localIdx )
   {
      const IndexType& sliceIndex = segmentsToSlicesMapping[ segmentIdx ];
      IndexType firstChunkOfSegment = 0;
      if( segmentIdx != slices[ sliceIndex ].firstSegment )
         firstChunkOfSegment = segmentsToChunksMapping[ segmentIdx - 1 ];

      // const IndexType lastChunkOfSegment = segmentsToChunksMapping[ segmentIdx ];
      // const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
      const IndexType sliceOffset = slices[ sliceIndex ].pointer;
      const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
      // TNL_ASSERT_LE( localIdx, segmentChunksCount * chunkSize, "" );
      TNL_ASSERT_LE( localIdx, ( segmentsToChunksMapping[ segmentIdx ] - firstChunkOfSegment ) * chunkSize, "" );

      if constexpr( Organization == RowMajorOrder )
         return sliceOffset + firstChunkOfSegment * chunkSize + localIdx;
      else {
         const IndexType inChunkOffset = localIdx % chunkSize;
         const IndexType chunkIdx = localIdx / chunkSize;
         return sliceOffset + inChunkOffset * chunksInSlice + firstChunkOfSegment + chunkIdx;
      }
   }

   [[nodiscard]] static IndexType
   getGlobalIndex( const ConstOffsetsView& segmentsToSlicesMapping,
                   const ConstSliceInfoContainerView& slices,
                   const ConstOffsetsView& segmentsToChunksMapping,
                   const IndexType chunksInSlice,
                   const IndexType segmentIdx,
                   const IndexType localIdx )
   {
      const IndexType& sliceIndex = segmentsToSlicesMapping.getElement( segmentIdx );
      IndexType firstChunkOfSegment = 0;
      if( segmentIdx != slices.getElement( sliceIndex ).firstSegment )
         firstChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx - 1 );

      // const IndexType lastChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx );
      // const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
      const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
      const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
      // TNL_ASSERT_LE( localIdx, segmentChunksCount * chunkSize, "" );
      TNL_ASSERT_LE( localIdx, ( segmentsToChunksMapping.getElement( segmentIdx ) - firstChunkOfSegment ) * chunkSize, "" );

      if constexpr( Organization == RowMajorOrder )
         return sliceOffset + firstChunkOfSegment * chunkSize + localIdx;
      else {
         const IndexType inChunkOffset = localIdx % chunkSize;
         const IndexType chunkIdx = localIdx / chunkSize;
         return sliceOffset + inChunkOffset * chunksInSlice + firstChunkOfSegment + chunkIdx;
      }
   }

   [[nodiscard]] __cuda_callable__
   static SegmentViewType
   getSegmentViewDirect( const ConstOffsetsView& segmentsToSlicesMapping,
                         const ConstSliceInfoContainerView& slices,
                         const ConstOffsetsView& segmentsToChunksMapping,
                         const IndexType& chunksInSlice,
                         const IndexType& segmentIdx )
   {
      const IndexType& sliceIndex = segmentsToSlicesMapping[ segmentIdx ];
      IndexType firstChunkOfSegment = 0;
      if( segmentIdx != slices[ sliceIndex ].firstSegment )
         firstChunkOfSegment = segmentsToChunksMapping[ segmentIdx - 1 ];

      const IndexType lastChunkOfSegment = segmentsToChunksMapping[ segmentIdx ];
      const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
      const IndexType sliceOffset = slices[ sliceIndex ].pointer;
      const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
      const IndexType segmentSize = segmentChunksCount * chunkSize;

      if constexpr( Organization == RowMajorOrder )
         return SegmentViewType(
            segmentIdx, sliceOffset + firstChunkOfSegment * chunkSize, segmentSize, chunkSize, chunksInSlice );
      else
         return SegmentViewType( segmentIdx, sliceOffset + firstChunkOfSegment, segmentSize, chunkSize, chunksInSlice );
   }

   [[nodiscard]] __cuda_callable__
   static SegmentViewType
   getSegmentView( const ConstOffsetsView& segmentsToSlicesMapping,
                   const ConstSliceInfoContainerView& slices,
                   const ConstOffsetsView& segmentsToChunksMapping,
                   const IndexType chunksInSlice,
                   const IndexType segmentIdx )
   {
      const IndexType& sliceIndex = segmentsToSlicesMapping.getElement( segmentIdx );
      IndexType firstChunkOfSegment = 0;
      if( segmentIdx != slices.getElement( sliceIndex ).firstSegment )
         firstChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx - 1 );

      const IndexType lastChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx );
      const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
      const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
      const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
      const IndexType segmentSize = segmentChunksCount * chunkSize;

      if constexpr( Organization == RowMajorOrder )
         return SegmentViewType(
            segmentIdx, sliceOffset + firstChunkOfSegment * chunkSize, segmentSize, chunkSize, chunksInSlice );
      else
         return SegmentViewType( segmentIdx, sliceOffset + firstChunkOfSegment, segmentSize, chunkSize, chunksInSlice );
   }
};

}  // namespace TNL::Algorithms::Segments::detail
