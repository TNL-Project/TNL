// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/VectorView.h>

#include "SegmentView.h"
#include "printSegments.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index >
class CSRView
{
public:
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   using OffsetsView = Containers::VectorView< Index, DeviceType, IndexType >;
   using ConstOffsetsView = typename OffsetsView::ConstViewType;
   using ViewType = CSRView;
   template< typename Device_, typename Index_ >
   using ViewTemplate = CSRView< Device_, Index_ >;
   using ConstViewType = CSRView< Device, std::add_const_t< Index > >;
   using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;

   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return false;
   }

   __cuda_callable__
   CSRView() = default;

   __cuda_callable__
   CSRView( const OffsetsView& offsets );

   __cuda_callable__
   CSRView( OffsetsView&& offsets );

   __cuda_callable__
   CSRView( const CSRView& csr_view ) = default;

   template< typename Index2 >
   __cuda_callable__
   CSRView( const CSRView< Device, Index2 >& csr_view );

   __cuda_callable__
   CSRView( CSRView&& csr_view ) noexcept = default;

   [[nodiscard]] static std::string
   getSerializationType();

   [[nodiscard]] static String
   getSegmentsType();

   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;

   /**
    * \brief Number segments.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentsCount() const;

   /***
    * \brief Returns size of the segment number \r segmentIdx
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   /***
    * \brief Returns number of elements managed by all segments.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const;

   /***
    * \brief Returns number of elements that needs to be allocated.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getStorageSize() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( Index segmentIdx, Index localIdx ) const;

   [[nodiscard]] __cuda_callable__
   SegmentViewType
   getSegmentView( IndexType segmentIdx ) const;

   /***
    * \brief Go over all segments and for each segment element call
    * function 'f'. The return type of 'f' is bool.
    * When its true, the for-loop continues. Once 'f' returns false, the for-loop
    * is terminated.
    */
   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   forAllElements( Function&& f ) const;

   template< typename Function >
   void
   forSegments( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   forAllSegments( Function&& f ) const;

   template< typename Function >
   void
   sequentialForSegments( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   sequentialForAllSegments( Function&& f ) const;

   CSRView&
   operator=( const CSRView& view );

   void
   save( File& file ) const;

   void
   load( File& file );

   [[nodiscard]] OffsetsView
   getOffsets()
   {
      return offsets;
   }

   [[nodiscard]] ConstOffsetsView
   getOffsets() const
   {
      return offsets.getConstView();
   }

protected:
   OffsetsView offsets;
};

template< typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str, const CSRView< Device, Index >& segments )
{
   return printSegments( str, segments );
}

}  // namespace TNL::Algorithms::Segments

#include "CSRView.hpp"
