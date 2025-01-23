// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>

namespace TNL::Algorithms::Segments {

template< typename Index, ElementsOrganization Organization >
class ChunkedEllpackSegmentView;

template< typename Index >
class ChunkedEllpackSegmentView< Index, ColumnMajorOrder >
{
public:
   using IndexType = Index;

   __cuda_callable__
   ChunkedEllpackSegmentView( IndexType segmentIdx, IndexType offset, IndexType size )
   : segmentIdx( segmentIdx ),
     segmentOffset( offset ),
     segmentSize( size )
   {}

   __cuda_callable__
   ChunkedEllpackSegmentView( const ChunkedEllpackSegmentView& ) = default;

   __cuda_callable__
   ChunkedEllpackSegmentView( ChunkedEllpackSegmentView&& ) noexcept = default;

   __cuda_callable__
   ChunkedEllpackSegmentView&
   operator=( const ChunkedEllpackSegmentView& ) = default;

   __cuda_callable__
   ChunkedEllpackSegmentView&
   operator=( ChunkedEllpackSegmentView&& ) noexcept = default;

   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const
   {
      return this->segmentSize;
   }

   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( IndexType localIndex ) const
   {
      TNL_ASSERT_LT( localIndex, segmentSize, "Local index exceeds segment bounds." );
      return segmentOffset + localIndex;
   }

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentIndex() const
   {
      return this->segmentIdx;
   }

protected:
   IndexType segmentIdx;
   IndexType segmentOffset;
   IndexType segmentSize;
};

template< typename Index >
class ChunkedEllpackSegmentView< Index, RowMajorOrder >
{
public:
   using IndexType = Index;

   __cuda_callable__
   ChunkedEllpackSegmentView( IndexType segmentIdx,
                              IndexType offset,
                              IndexType size,
                              IndexType chunkSize,
                              IndexType chunksInSlice )
   : segmentIdx( segmentIdx ),
     segmentOffset( offset ),
     segmentSize( size ),
     chunkSize( chunkSize ),
     chunksInSlice( chunksInSlice )
   {}

   __cuda_callable__
   ChunkedEllpackSegmentView( const ChunkedEllpackSegmentView& ) = default;

   __cuda_callable__
   ChunkedEllpackSegmentView( ChunkedEllpackSegmentView&& ) noexcept = default;

   __cuda_callable__
   ChunkedEllpackSegmentView&
   operator=( const ChunkedEllpackSegmentView& ) = default;

   __cuda_callable__
   ChunkedEllpackSegmentView&
   operator=( ChunkedEllpackSegmentView&& ) noexcept = default;

   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const
   {
      return this->segmentSize;
   }

   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( IndexType localIdx ) const
   {
      TNL_ASSERT_LT( localIdx, segmentSize, "Local index exceeds segment bounds." );
      const IndexType chunkIdx = localIdx / chunkSize;
      const IndexType inChunkOffset = localIdx % chunkSize;
      return segmentOffset + inChunkOffset * chunksInSlice + chunkIdx;
   }

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentIndex() const
   {
      return this->segmentIdx;
   }

protected:
   IndexType segmentIdx;
   IndexType segmentOffset;
   IndexType segmentSize;
   IndexType chunkSize;
   IndexType chunksInSlice;
};

}  // namespace TNL::Algorithms::Segments
