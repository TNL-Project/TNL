// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/Segments/SegmentViewIterator.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for accessing particular segment.
 *
 * \tparam Index is type for indexing elements in related segments.
 *
 * See the template specializations \ref TNL::Algorithms::Segments::SegmentView< Index, ColumnMajorOrder >
 *  and \ref TNL::Algorithms::Segments::SegmentView< Index, RowMajorOrder > for column-major
 * and row-major elements organization respectively. They have equivalent interface.
 */
template< typename Index, ElementsOrganization Organization >
class SegmentView;

/**
 * \brief Data structure for accessing particular segment.
 *
 * \tparam Index is type for indexing elements in related segments.
 */
template< typename Index >
class SegmentView< Index, ColumnMajorOrder >
{
public:
   //! \brief Type for indexing elements in related segments.
   using IndexType = Index;

   //! \brief Type of iterator for iterating over elements of the segment.
   using IteratorType = SegmentViewIterator< SegmentView >;

   /**
    * \brief Constructor with all parameters.
    *
    * \param segmentIdx is an index of segment the segment view will point to.
    * \param offset is an offset of the segment in the parent segments.
    * \param size is a size of the segment.
    * \param step is stepping between neighbouring elements in the segment.
    */
   __cuda_callable__
   SegmentView( IndexType segmentIdx, IndexType offset, IndexType size, IndexType step )
   : segmentIdx( segmentIdx ),
     segmentOffset( offset ),
     segmentSize( size ),
     step( step )
   {}

   //! \brief Copy constructor.
   __cuda_callable__
   SegmentView( const SegmentView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   SegmentView( SegmentView&& ) noexcept = default;

   //! \brief Copy assignment operator.
   __cuda_callable__
   SegmentView&
   operator=( const SegmentView& ) = default;

   //! \brief Move assignment operator.
   __cuda_callable__
   SegmentView&
   operator=( SegmentView&& ) noexcept = default;

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
      return segmentOffset + localIndex * step;
   }

   /**
    * \brief Set index of the segment.
    *
    * \param index is the of the segment.
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

   /**
    * \brief Returns iterator pointing at the beginning of the segment.
    *
    * \return iterator pointing at the beginning.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   begin() const
   {
      return IteratorType( *this, 0 );
   }

   /**
    * \brief Returns iterator pointing at the end of the segment.
    *
    * \return iterator pointing at the end.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   end() const
   {
      return IteratorType( *this, this->getSize() );
   }

   /**
    * \brief Returns constant iterator pointing at the beginning of the segment.
    *
    * \return iterator pointing at the beginning.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   cbegin() const
   {
      return IteratorType( *this, 0 );
   }

   /**
    * \brief Returns constant iterator pointing at the end of the segment.
    *
    * \return iterator pointing at the end.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   cend() const
   {
      return IteratorType( *this, this->getSize() );
   }

protected:
   IndexType segmentIdx;
   IndexType segmentOffset;
   IndexType segmentSize;
   IndexType step;
};

template< typename Index >
class SegmentView< Index, RowMajorOrder >
{
public:
   /**
    * \brief Type for indexing elements in related segments.
    */
   using IndexType = Index;

   /**
    * \brief Type of iterator for iterating over elements of the segment.
    */
   using IteratorType = SegmentViewIterator< SegmentView >;

   /**
    * \brief Constructor with all parameters.
    *
    * \param segmentIdx is an index of segment the segment view will point to.
    * \param offset is an offset of the segment in the parent segments.
    * \param size is a size of the segment.
    */
   __cuda_callable__
   SegmentView( IndexType segmentIdx, IndexType offset, IndexType size )
   : segmentIdx( segmentIdx ),
     segmentOffset( offset ),
     segmentSize( size )
   {}

   /**
    * \brief Copy constructor.
    */
   __cuda_callable__
   SegmentView( const SegmentView& ) = default;

   /**
    * \brief Move constructor.
    */
   __cuda_callable__
   SegmentView( SegmentView&& ) noexcept = default;

   /**
    * \brief Copy assignment operator.
    */
   __cuda_callable__
   SegmentView&
   operator=( const SegmentView& ) = default;

   /**
    * \brief Move assignment operator.
    */
   __cuda_callable__
   SegmentView&
   operator=( SegmentView&& ) noexcept = default;

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

   /**
    * \brief Set index of the segment.
    *
    * \param index is index of the segment.
    */
   __cuda_callable__
   void
   setSegmentIndex( IndexType index )
   {
      this->segmentIdx = index;
   }

   /**
    * \brief Returns iterator pointing at the beginning of the segment.
    *
    * \return iterator pointing at the beginning.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   begin() const
   {
      return IteratorType( *this, 0 );
   }

   /**
    * \brief Returns iterator pointing at the end of the segment.
    *
    * \return iterator pointing at the end.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   end() const
   {
      return IteratorType( *this, this->getSize() );
   }

   /**
    * \brief Returns constant iterator pointing at the beginning of the segment.
    *
    * \return iterator pointing at the beginning.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   cbegin() const
   {
      return IteratorType( *this, 0 );
   }

   /**
    * \brief Returns constant iterator pointing at the end of the segment.
    *
    * \return iterator pointing at the end.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   cend() const
   {
      return IteratorType( *this, this->getSize() );
   }

protected:
   IndexType segmentIdx;
   IndexType segmentOffset;
   IndexType segmentSize;
};

}  // namespace TNL::Algorithms::Segments
