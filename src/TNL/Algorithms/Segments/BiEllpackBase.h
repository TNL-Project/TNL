// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/VectorView.h>

#include "ElementsOrganization.h"
#include "BiEllpackSegmentView.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e BiEllpackBase serves as a base class for \ref TNL::Algorithms::Segments::BiEllpack and \ref
 * TNL::Algorithms::Segments::BiEllpackView.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam Organization is the organization of the elements in the segments—either row-major or column-major order.
 * \tparam WarpSize is the warp size used for the segments.
 */
template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize >
class BiEllpackBase
{
public:
   //! \brief The device where the segments are operating.
   using DeviceType = Device;

   //! \brief The type used for indexing of segments elements.
   using IndexType = std::remove_const_t< Index >;

   //! \brief The type for representing the vector view with offsets of the BiEllpack groups.
   using OffsetsView = Containers::VectorView< Index, DeviceType, IndexType >;

   //! \brief The type for representing the constant vector view with offsets of the BiEllpack groups.
   using ConstOffsetsView = typename OffsetsView::ConstViewType;

   //! \brief Accessor type for one particular segment.
   using SegmentViewType = BiEllpackSegmentView< IndexType, Organization, WarpSize >;

   //! \brief Returns the warp size used for the segments.
   [[nodiscard]] static constexpr int
   getWarpSize()
   {
      return WarpSize;
   }

   //! \brief Returns the log2 of the warp size used for the segments.
   [[nodiscard]] static constexpr int
   getLogWarpSize()
   {
      return TNL::discreteLog2( WarpSize );
   }

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
   BiEllpackBase() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   BiEllpackBase( const BiEllpackBase& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   BiEllpackBase( BiEllpackBase&& ) noexcept = default;

   //! \brief Constructor that initializes segments based all necessary data.
   __cuda_callable__
   BiEllpackBase( IndexType size, IndexType storageSize, OffsetsView segmentsPermutation, OffsetsView groupPointers );

   //! \brief Copy-assignment operator.
   BiEllpackBase&
   operator=( const BiEllpackBase& ) = delete;

   //! \brief Move-assignment operator.
   BiEllpackBase&
   operator=( BiEllpackBase&& ) = delete;

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
   getSegmentCount() const;

   //! \brief Returns the number of segments. Deprecated, use getSegmentCount().
   [[deprecated( "Use getSegmentCount() instead." )]] [[nodiscard]] __cuda_callable__
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
   getGlobalIndex( Index segmentIdx, Index localIdx ) const;

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

   //! \brief Returns a view containing the permutation of segments.
   [[nodiscard]] __cuda_callable__
   OffsetsView
   getSegmentsPermutationView();

   //! \brief Returns a view containing the permutation of segments.
   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getSegmentsPermutationView() const;

   //! \brief Returns a view containing the pointers to the beginning of each segment.
   [[nodiscard]] __cuda_callable__
   OffsetsView
   getGroupPointersView();

   //! \brief Returns a view containing the pointers to the beginning of each segment.
   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getGroupPointersView() const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getVirtualSegments() const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead" )]] void
   forElements( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllElements instead" )]] void
   forAllElements( Function&& function ) const;

   template< typename Array, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead" )]] void
   forElements( const Array& segmentIndexes, Index begin, Index end, Function function ) const;

   template< typename Array, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElements instead" )]] void
   forElements( const Array& segmentIndexes, Function function ) const;

   template< typename Condition, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forElementsIf instead" )]] void
   forElementsIf( IndexType begin, IndexType end, Condition condition, Function function ) const;

   template< typename Condition, typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllElementsIf instead" )]] void
   forAllElementsIf( Condition condition, Function function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forSegments instead" )]] void
   forSegments( IndexType begin, IndexType end, Function&& function ) const;

   template< typename Function >
   [[deprecated( "Use TNL::Algorithms::Segments::forAllSegments instead" )]] void
   forAllSegments( Function&& function ) const;

   //! \brief Prints the structure of the segments to the specified output stream.
   void
   printStructure( std::ostream& str ) const;

protected:
   IndexType size = 0;
   IndexType storageSize = 0;
   OffsetsView segmentsPermutation;
   OffsetsView groupPointers;

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
   bind( IndexType size, IndexType storageSize, OffsetsView segmentsPermutation, OffsetsView groupPointers );

   [[nodiscard]] __cuda_callable__
   IndexType
   getVirtualSegments( IndexType segmentsCount ) const;
};

}  // namespace TNL::Algorithms::Segments

#include "BiEllpackBase.hpp"
#include "print.h"
