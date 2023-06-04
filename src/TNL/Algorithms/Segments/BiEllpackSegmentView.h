// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL::Algorithms::Segments {

template< typename Index, ElementsOrganization Organization, int WarpSize = 32 >
class BiEllpackSegmentView
{
public:
   [[nodiscard]] static constexpr int
   getWarpSize()
   {
      return WarpSize;
   }

   [[nodiscard]] static constexpr int
   getLogWarpSize()
   {
      return TNL::discreteLog2( WarpSize );
   }

   [[nodiscard]] static constexpr int
   getGroupsCount()
   {
      return getLogWarpSize() + 1;
   }

   using IndexType = Index;
   using GroupsWidthType = Containers::StaticVector< getGroupsCount(), IndexType >;

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
   : segmentIdx( segmentIdx ), groupOffset( offset ), inStripIdx( inStripIdx ), segmentSize( TNL::sum( groupsWidth ) ),
     groupsWidth( groupsWidth )
   {}

   __cuda_callable__
   BiEllpackSegmentView( const BiEllpackSegmentView& ) = default;

   __cuda_callable__
   BiEllpackSegmentView( BiEllpackSegmentView&& ) noexcept = default;

   __cuda_callable__
   BiEllpackSegmentView&
   operator=( const BiEllpackSegmentView& ) = default;

   __cuda_callable__
   BiEllpackSegmentView&
   operator=( BiEllpackSegmentView&& ) noexcept = default;

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
      // std::cerr << "SegmentView: localIdx = " << localIdx << " groupWidth = " << groupsWidth << std::endl;
      IndexType groupIdx = 0;
      IndexType offset = groupOffset;
      IndexType groupHeight = getWarpSize();
      while( localIdx >= groupsWidth[ groupIdx ] ) {
         // std::cerr << "ROW: groupIdx = " << groupIdx << " groupWidth = " << groupsWidth[ groupIdx ]
         //           << " groupSize = " << groupsWidth[ groupIdx ] * groupHeight << std::endl;
         localIdx -= groupsWidth[ groupIdx ];
         offset += groupsWidth[ groupIdx++ ] * groupHeight;
         groupHeight /= 2;
      }
      TNL_ASSERT_LE( groupIdx, TNL::log2( getWarpSize() - inStripIdx + 1 ), "Local index exceeds segment bounds." );
      if constexpr( Organization == RowMajorOrder ) {
         // std::cerr << " offset = " << offset << " inStripIdx = " << inStripIdx << " localIdx = " << localIdx
         //           << " return = " << offset + inStripIdx * groupsWidth[ groupIdx ] + localIdx << std::endl;
         return offset + inStripIdx * groupsWidth[ groupIdx ] + localIdx;
      }
      else
         return offset + inStripIdx + localIdx * groupHeight;
   }

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
