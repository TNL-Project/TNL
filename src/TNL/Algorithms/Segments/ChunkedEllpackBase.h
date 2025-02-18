// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/VectorView.h>

#include "ElementsOrganization.h"
#include "ChunkedEllpackSegmentView.h"
#include "detail/ChunkedEllpack.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization >
class ChunkedEllpackBase
{
public:
   using DeviceType = Device;

   using IndexType = std::remove_const_t< Index >;

   using OffsetsView = Containers::VectorView< Index, DeviceType, IndexType >;

   using ConstOffsetsView = typename OffsetsView::ConstViewType;

   template< typename Device_, typename Index_ >
   using ViewTemplate = ChunkedEllpackBase< Device_, Index_, Organization >;

   using SegmentViewType = ChunkedEllpackSegmentView< IndexType, Organization >;

   using SliceInfoType = detail::ChunkedEllpackSliceInfo< IndexType >;
   using SliceInfoContainerView =
      Containers::ArrayView< typename TNL::copy_const< SliceInfoType >::template from< Index >::type, DeviceType, IndexType >;
   using ConstSliceInfoContainerView = typename SliceInfoContainerView::ConstViewType;

   [[nodiscard]] static constexpr ElementsOrganization
   getOrganization()
   {
      return Organization;
   }

   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return true;
   }

   __cuda_callable__
   ChunkedEllpackBase() = default;

   __cuda_callable__
   ChunkedEllpackBase( IndexType size,
                       IndexType storageSize,
                       IndexType numberOfSlices,
                       IndexType chunksInSlice,
                       IndexType desiredChunkSize,
                       OffsetsView segmentToChunkMapping,
                       OffsetsView segmentToSliceMapping,
                       OffsetsView chunksToSegmentsMapping,
                       OffsetsView segmentPointers,
                       SliceInfoContainerView slices );

   __cuda_callable__
   ChunkedEllpackBase( const ChunkedEllpackBase& ) = default;

   __cuda_callable__
   ChunkedEllpackBase( ChunkedEllpackBase&& ) noexcept = default;

   ChunkedEllpackBase&
   operator=( const ChunkedEllpackBase& ) = delete;

   ChunkedEllpackBase&
   operator=( ChunkedEllpackBase&& ) = delete;

   [[nodiscard]] static std::string
   getSerializationType();

   [[nodiscard]] static std::string
   getSegmentsType();

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentsCount() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getStorageSize() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( IndexType segmentIdx, IndexType localIdx ) const;

   [[nodiscard]] __cuda_callable__
   SegmentViewType
   getSegmentView( IndexType segmentIdx ) const;

   [[nodiscard]] __cuda_callable__
   OffsetsView
   getSegmentToChunkMappingView();

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSegmentToChunkMappingView() const;

   [[nodiscard]] __cuda_callable__
   OffsetsView
   getSegmentToSliceMappingView();

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSegmentToSliceMappingView() const;

   [[nodiscard]] __cuda_callable__
   OffsetsView
   getChunksToSegmentsMappingView();

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getChunksToSegmentsMappingView() const;

   [[nodiscard]] __cuda_callable__
   OffsetsView
   getSegmentPointersView();

   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSegmentPointersView() const;

   [[nodiscard]] __cuda_callable__
   SliceInfoContainerView
   getSlicesView();

   [[nodiscard]] __cuda_callable__
   ConstSliceInfoContainerView
   getSlicesView() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getNumberOfSlices() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getChunksInSlice() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getDesiredChunkSize() const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead." )]]
   void
   forElements( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllElements instead." )]]
   void
   forAllElements( Function&& function ) const;

   template< typename Array, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead." )]]
   void
   forElements( const Array& segmentIndexes, Index begin, Index end, Function function ) const;

   template< typename Array, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead." )]]
   void
   forElements( const Array& segmentIndexes, Function function ) const;

   template< typename Condition, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElementsIf instead." )]]
   void
   forElementsIf( IndexType begin, IndexType end, Condition condition, Function function ) const;

   template< typename Condition, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllElementsIf instead." )]]
   void
   forAllElementsIf( Condition condition, Function function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forSegments instead." )]]
   void
   forSegments( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllSegments instead." )]]
   void
   forAllSegments( Function&& function ) const;

   // TODO: sequentialForSegments, sequentialForAllSegments

   void
   printStructure( std::ostream& str ) const;

protected:
   IndexType size = 0;
   IndexType storageSize = 0;
   IndexType numberOfSlices = 0;
   IndexType chunksInSlice = 256;
   IndexType desiredChunkSize = 16;

   //! \brief For each segment, this keeps index of the first chunk within a slice.
   OffsetsView segmentToChunkMapping;

   //! \brief For each segment, this keeps index of the slice which contains the segment.
   OffsetsView segmentToSliceMapping;

   OffsetsView chunksToSegmentsMapping;

   //! \brief Keeps index of the first segment index.
   OffsetsView segmentPointers;

   SliceInfoContainerView slices;

   /**
    * \brief Re-initializes the internal attributes of the base class.
    *
    * Note that this function is \e protected to ensure that the user cannot
    * modify the base class of segments. For the same reason, in future code
    * development we also need to make sure that all non-const functions in
    * the base class return by value and not by reference.
    */
   __cuda_callable__
   void
   bind( IndexType size,
         IndexType storageSize,
         IndexType numberOfSlices,
         IndexType chunksInSlice,
         IndexType desiredChunkSize,
         OffsetsView segmentToChunkMapping,
         OffsetsView segmentToSliceMapping,
         OffsetsView chunksToSegmentsMapping,
         OffsetsView segmentPointers,
         SliceInfoContainerView slices );
};

}  // namespace TNL::Algorithms::Segments

#include "ChunkedEllpackBase.hpp"
#include "print.h"
