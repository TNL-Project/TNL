// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for accessing particular segment of BiEllpack segments.
 *
 * \tparam Index is type for indexing elements in related segments.
 * \tparam Organization is the organization of the elements in the segments—either row-major or column-major order.
 * \tparam WarpSize is the warp size used for the segments.
 *
 * See the template specializations \ref TNL::Algorithms::Segments::SegmentView< Index, ColumnMajorOrder >
 *  and \ref TNL::Algorithms::Segments::SegmentView< Index, RowMajorOrder > for column-major
 * and row-major elements organization respectively. They have equivalent interface.
 */
template< typename Index, ElementsOrganization Organization, int WarpSize = Backend::getWarpSize() >
class BiEllpackSegmentView
{
public:
   //! \brief Type for indexing elements in related segments.
   using IndexType = Index;

   //! \brief Returns the warp size used for the segments.
   [[nodiscard]] static constexpr int
   getWarpSize()
   {
      return WarpSize;
   }

   //! \brief Returns the log of the warp size used for the segments.
   [[nodiscard]] static constexpr int
   getLogWarpSize()
   {
      return TNL::discreteLog2( WarpSize );
   }

   //! \brief Returns the number of groups used for the segments.
   [[nodiscard]] static constexpr int
   getGroupsCount()
   {
      return getLogWarpSize() + 1;
   }

   using GroupsWidthType = Containers::StaticVector< getGroupsCount(), IndexType >;

   //! \brief Copy constructor.
   __cuda_callable__
   BiEllpackSegmentView( const BiEllpackSegmentView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   BiEllpackSegmentView( BiEllpackSegmentView&& ) noexcept = default;

   /**
    * \brief Constructor.
    *
    * \param segmentIdx is the segment index.
    * \param offset is offset of the first group of the strip the segment belongs to.
    * \param inStripIdx is index of the segment within its strip.
    * \param groupsWidth is a static vector containing widths of the strip groups.
    */
   __cuda_callable__
   BiEllpackSegmentView( IndexType segmentIdx, IndexType offset, IndexType inStripIdx, const GroupsWidthType& groupsWidth )
   : segmentIdx( segmentIdx ),
     groupOffset( offset ),
     inStripIdx( inStripIdx ),
     segmentSize( TNL::sum( groupsWidth ) ),
     groupsWidth( groupsWidth )
   {}

   //! \brief Copy assignment operator.
   __cuda_callable__
   BiEllpackSegmentView&
   operator=( const BiEllpackSegmentView& ) = default;

   //! \brief Move assignment operator.
   __cuda_callable__
   BiEllpackSegmentView&
   operator=( BiEllpackSegmentView&& ) noexcept = default;

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
      IndexType groupIdx = 0;
      IndexType offset = groupOffset;
      IndexType groupHeight = getWarpSize();
      while( localIdx >= groupsWidth[ groupIdx ] ) {
         localIdx -= groupsWidth[ groupIdx ];
         offset += groupsWidth[ groupIdx++ ] * groupHeight;
         groupHeight /= 2;
      }
      TNL_ASSERT_LE( groupIdx, TNL::log2( getWarpSize() - inStripIdx + 1 ), "Local index exceeds segment bounds." );
      if constexpr( Organization == RowMajorOrder ) {
         return offset + inStripIdx * groupsWidth[ groupIdx ] + localIdx;
      }
      else
         return offset + inStripIdx + localIdx * groupHeight;
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
   IndexType groupOffset;
   IndexType inStripIdx;
   IndexType segmentSize;
   GroupsWidthType groupsWidth;
};

}  // namespace TNL::Algorithms::Segments
