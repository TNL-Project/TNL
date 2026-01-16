// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/VectorView.h>

#include "ElementsOrganization.h"
#include "ChunkedEllpackSegmentView.h"
#include "detail/ChunkedEllpack.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e ChunkedEllpackBase serves as a base class for \ref TNL::Algorithms::Segments::ChunkedEllpack and \ref
 * TNL::Algorithms::Segments::ChunkedEllpackView.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam Organization is the organization of the elements in the segments—either row-major or column-major order.
 */
template< typename Device, typename Index, ElementsOrganization Organization >
class ChunkedEllpackBase
{
public:
   //! \brief The device where the segments are operating.
   using DeviceType = Device;

   //! \brief The type used for indexing of segments elements.
   using IndexType = std::remove_const_t< Index >;

   //! \brief The type for representing the vector view with segment offsets.
   using OffsetsView = Containers::VectorView< Index, DeviceType, IndexType >;

   //! \brief The type for representing the constant vector view with segment offsets.
   using ConstOffsetsView = typename OffsetsView::ConstViewType;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = ChunkedEllpackBase< Device_, Index_, Organization >;

   //! \brief Accessor type for one particular segment.
   using SegmentViewType = ChunkedEllpackSegmentView< IndexType, Organization >;

   using SliceInfoType = detail::ChunkedEllpackSliceInfo< IndexType >;
   using SliceInfoContainerView =
      Containers::ArrayView< typename TNL::copy_const< SliceInfoType >::template from< Index >::type, DeviceType, IndexType >;
   using ConstSliceInfoContainerView = typename SliceInfoContainerView::ConstViewType;

   //! \brief Returns the data layout for the CSR format (it is always row-major order).
   [[nodiscard]] static constexpr ElementsOrganization
   getOrganization()
   {
      return Organization;
   }

   //! \brief This function denotes that the CSR format does not use padding elements.
   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return true;
   }

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   ChunkedEllpackBase() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   ChunkedEllpackBase( const ChunkedEllpackBase& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   ChunkedEllpackBase( ChunkedEllpackBase&& ) noexcept = default;

   //! \brief Constructor with all necessary data and views.
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

   //! \brief Copy-assignment operator.
   ChunkedEllpackBase&
   operator=( const ChunkedEllpackBase& ) = delete;

   //! \brief Move-assignment operator.
   ChunkedEllpackBase&
   operator=( ChunkedEllpackBase&& ) = delete;

   /**
    * \brief Returns string with the serialization type.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_getSerializationType.cpp
    * \par Output
    * \include SegmentsExample_getSerializationType.out
    */
   [[nodiscard]] static std::string
   getSerializationType();

   /**
    * \brief Returns string with the segments type.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_getSegmentsType.cpp
    * \par Output
    * \include SegmentsExample_getSegmentsType.out
    */
   [[nodiscard]] static std::string
   getSegmentsType();

   //! \brief Returns the number of segments.
   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentsCount() const;

   //! \brief Returns the size of a particular segment denoted by \e segmentIdx.
   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   //! \brief Returns the number of elements managed by all segments.
   [[deprecated( "Use getElementCount() instead." )]] [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const;

   //! \brief Returns the number of elements managed by all segments.
   [[nodiscard]] __cuda_callable__
   IndexType
   getElementCount() const;

   //! \brief Returns number of elements that needs to be allocated by a
   //! container connected to this segments.
   [[nodiscard]] __cuda_callable__
   IndexType
   getStorageSize() const;

   /**
    * \brief Computes the global index of an element managed by the segments.
    *
    * The global index serves as a reference to the element within its container.
    *
    * \param segmentIdx The index of the segment containing the element.
    * \param localIdx The local index of the element within the segment.
    * \return The global index of the element.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( IndexType segmentIdx, IndexType localIdx ) const;

   /**
    * \brief Returns a segment view (i.e., a segment accessor) for the specified segment index.
    *
    * \param segmentIdx The index of the requested segment.
    * \return The segment view of the specified segment.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_getSegmentView.cpp
    * \par Output
    * \include SegmentsExample_getSegmentView.out
    */
   [[nodiscard]] __cuda_callable__
   SegmentViewType
   getSegmentView( IndexType segmentIdx ) const;

   //! \brief Returns a modifiable vector view with mapping of segments to chunks.
   [[nodiscard]] __cuda_callable__
   OffsetsView
   getSegmentToChunkMappingView();

   //! \brief Returns a constant vector view with mapping of segments to chunks.
   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSegmentToChunkMappingView() const;

   //! \brief Returns a modifiable vector view with mapping of segments to slices.
   [[nodiscard]] __cuda_callable__
   OffsetsView
   getSegmentToSliceMappingView();

   //! \brief Returns a constant vector view with mapping of segments to slices.
   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSegmentToSliceMappingView() const;

   //! \brief Returns a modifiable vector view with mapping of chunks to segments.
   [[nodiscard]] __cuda_callable__
   OffsetsView
   getChunksToSegmentsMappingView();

   //! \brief Returns a constant vector view with mapping of chunks to segments.
   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getChunksToSegmentsMappingView() const;

   //! \brief Returns a modifiable vector view with segment pointers.
   [[nodiscard]] __cuda_callable__
   OffsetsView
   getSegmentPointersView();

   //! \brief Returns a constant vector view with segment pointers.
   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSegmentPointersView() const;

   //! \brief Returns a modifiable view with slice information.
   [[nodiscard]] __cuda_callable__
   SliceInfoContainerView
   getSlicesView();

   //! \brief Returns a constant view with slice information.
   [[nodiscard]] __cuda_callable__
   ConstSliceInfoContainerView
   getSlicesView() const;

   //! \brief Returns the number of slices.
   [[nodiscard]] __cuda_callable__
   IndexType
   getNumberOfSlices() const;

   //! \brief Returns the number of chunks in a slice.
   [[nodiscard]] __cuda_callable__
   IndexType
   getChunksInSlice() const;

   //! \brief Returns the desired chunk size.
   [[nodiscard]] __cuda_callable__
   IndexType
   getDesiredChunkSize() const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead." )]] void
   forElements( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllElements instead." )]] void
   forAllElements( Function&& function ) const;

   template< typename Array, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead." )]] void
   forElements( const Array& segmentIndexes, Index begin, Index end, Function function ) const;

   template< typename Array, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead." )]] void
   forElements( const Array& segmentIndexes, Function function ) const;

   template< typename Condition, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElementsIf instead." )]] void
   forElementsIf( IndexType begin, IndexType end, Condition condition, Function function ) const;

   template< typename Condition, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllElementsIf instead." )]] void
   forAllElementsIf( Condition condition, Function function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forSegments instead." )]] void
   forSegments( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllSegments instead." )]] void
   forAllSegments( Function&& function ) const;

   //! \brief Prints the structure of the segments to the output stream.
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
