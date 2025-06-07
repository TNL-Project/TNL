// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>

namespace TNL::Algorithms::Segments {

template< typename Index, ElementsOrganization Organization >
class ChunkedEllpackSegmentView;

/**
 * \brief Data structure for accessing particular segment of row-major Chunked Ellpack segments.
 *
 * \tparam Index is type for indexing elements in related segments.
 *
 * See the template specializations \ref TNL::Algorithms::Segments::SegmentView< Index, ColumnMajorOrder >
 *  and \ref TNL::Algorithms::Segments::SegmentView< Index, RowMajorOrder > for column-major
 * and row-major elements organization respectively. They have equivalent interface.
 */
template< typename Index >
class ChunkedEllpackSegmentView< Index, RowMajorOrder >
{
public:
   //! \brief Type for indexing elements in related segments.
   using IndexType = Index;

   //! \brief Constructor with all parameters.
   __cuda_callable__
   ChunkedEllpackSegmentView( IndexType segmentIdx, IndexType offset, IndexType size )
   : segmentIdx( segmentIdx ),
     segmentOffset( offset ),
     segmentSize( size )
   {}

   //! \brief Copy constructor.
   __cuda_callable__
   ChunkedEllpackSegmentView( const ChunkedEllpackSegmentView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   ChunkedEllpackSegmentView( ChunkedEllpackSegmentView&& ) noexcept = default;

   //! \brief Copy assignment operator.
   __cuda_callable__
   ChunkedEllpackSegmentView&
   operator=( const ChunkedEllpackSegmentView& ) = default;

   //! \brief Move assignment operator.
   __cuda_callable__
   ChunkedEllpackSegmentView&
   operator=( ChunkedEllpackSegmentView&& ) noexcept = default;

   /**
    * \brief Get the size of the segment, i.e. number of elements in the segment.
    *
    * \return number of elements in the segment.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const
   {
      return this->segmentSize;
   }

   /**
    * \brief Get global index of an element with rank \e localIndex in the segment.
    *
    * \param localIndex is the rank of the element in the segment.
    * \return global index of the element.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( IndexType localIndex ) const
   {
      TNL_ASSERT_LT( localIndex, segmentSize, "Local index exceeds segment bounds." );
      return segmentOffset + localIndex;
   }

   /**
    * \brief Set index of the segment.
    *
    * \param index of the segment.
    */
   __cuda_callable__
   void
   setSegmentIndex( IndexType index )
   {
      this->segmentIdx = index;
   }

   /**
    * \brief Get index of the segment.
    *
    * \return index of the segment.
    */
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

/**
 * \brief Data structure for accessing particular segment of column-major Chunked Ellpack segments.
 *
 * \tparam Index is type for indexing elements in related segments.
 *
 * See the template specializations \ref TNL::Algorithms::Segments::SegmentView< Index, ColumnMajorOrder >
 *  and \ref TNL::Algorithms::Segments::SegmentView< Index, RowMajorOrder > for column-major
 * and row-major elements organization respectively. They have equivalent interface.
 */
template< typename Index >
class ChunkedEllpackSegmentView< Index, ColumnMajorOrder >
{
public:
   //! \brief Type for indexing elements in related segments.
   using IndexType = Index;

   //! \brief Constructor with all parameters.
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

   //! \brief Copy constructor.
   __cuda_callable__
   ChunkedEllpackSegmentView( const ChunkedEllpackSegmentView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   ChunkedEllpackSegmentView( ChunkedEllpackSegmentView&& ) noexcept = default;

   //! \brief Copy assignment operator.
   __cuda_callable__
   ChunkedEllpackSegmentView&
   operator=( const ChunkedEllpackSegmentView& ) = default;

   //! \brief Move assignment operator.
   __cuda_callable__
   ChunkedEllpackSegmentView&
   operator=( ChunkedEllpackSegmentView&& ) noexcept = default;

   /**
    * \brief Get the size of the segment, i.e. number of elements in the segment.
    *
    * \return number of elements in the segment.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const
   {
      return this->segmentSize;
   }

   /**
    * \brief Get global index of an element with rank \e localIdx in the segment.
    *
    * \param localIdx is the rank of the element in the segment.
    * \return global index of the element.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( IndexType localIdx ) const
   {
      TNL_ASSERT_LT( localIdx, segmentSize, "Local index exceeds segment bounds." );
      const IndexType chunkIdx = localIdx / chunkSize;
      const IndexType inChunkOffset = localIdx % chunkSize;
      return segmentOffset + inChunkOffset * chunksInSlice + chunkIdx;
   }

   /**
    * \brief Set index of the segment.
    *
    * \param index of the segment.
    */
   __cuda_callable__
   void
   setSegmentIndex( IndexType index )
   {
      this->segmentIdx = index;
   }

   /**
    * \brief Get index of the segment.
    *
    * \return index of the segment.
    */
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
