// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/VectorView.h>

#include "SegmentView.h"
#include "printSegments.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e CSRBase serves as a base class for \ref CSR and \ref CSRView.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 */
template< typename Device, typename Index >
class CSRBase
{
public:
   //! \brief The device where the segments are operating.
   using DeviceType = Device;

   //! \brief The type used for indexing of segments elements.
   using IndexType = std::remove_const_t< Index >;

   //! \brief The type for representing the vector view with row offsets used
   //! in the CSR format.
   using OffsetsView = Containers::VectorView< Index, DeviceType, IndexType >;

   //! \brief The type for representing the constant vector view with row
   //! offsets used in the CSR format.
   using ConstOffsetsView = typename OffsetsView::ConstViewType;

   //! \brief Accessor type fro one particular segment.
   using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;

   //! \brief Returns the data layout for the CSR format (it is always row-major order).
   [[nodiscard]] static constexpr ElementsOrganization
   getOrganization()
   {
      return RowMajorOrder;
   }

   //! \brief This function denotes that the CSR format does not use padding elements.
   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return false;
   }

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   CSRBase() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   CSRBase( const CSRBase& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   CSRBase( CSRBase&& ) noexcept = default;

   //! \brief Binds a new CSR view to an offsets vector.
   __cuda_callable__
   CSRBase( const OffsetsView& offsets );

   //! \brief Binds a new CSR view to an offsets vector.
   __cuda_callable__
   CSRBase( OffsetsView&& offsets );

   //! \brief Copy-assignment operator.
   CSRBase&
   operator=( const CSRBase& ) = delete;

   //! \brief Move-assignment operator.
   CSRBase&
   operator=( CSRBase&& ) = delete;

   /**
    * \brief Returns string with the serialization type.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_getSerializationType.cpp
    * \par Output
    * \include SegmentsExample_CSR_getSerializationType.out
    */
   [[nodiscard]] static std::string
   getSerializationType();

   /**
    * \brief Returns string with the segments type.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_getSegmentsType.cpp
    * \par Output
    * \include SegmentsExample_CSR_getSegmentsType.out
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
   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const;

   //! \brief Returns number of elements that needs to be allocated by a
   //! container connected to this segments.
   [[nodiscard]] __cuda_callable__
   IndexType
   getStorageSize() const;

   /**
    * \brief Computes the global index of an element managed by the segments.
    *
    * The global index serves as a refernce on the element in its container.
    *
    * \param segmentIdx is index of a segment with the element.
    * \param localIdx is tha local index of the element within the segment.
    * \return global index of the element.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( Index segmentIdx, Index localIdx ) const;

   /**
    * \brief Returns segment view (i.e. segment accessor) of segment with given
    * index.
    *
    * \param segmentIdx is index of the request segment.
    * \return segment view of given segment.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_getSegmentView.cpp
    * \par Output
    * \include SegmentsExample_CSR_getSegmentView.out
    */
   [[nodiscard]] __cuda_callable__
   SegmentViewType
   getSegmentView( IndexType segmentIdx ) const;

   //! \brief Returns a modifiable vector view with row offsets used in the CSR format.
   [[nodiscard]] __cuda_callable__
   OffsetsView
   getOffsets();

   //! \brief Returns a constant vector view with row offsets used in the CSR format.
   [[nodiscard]] __cuda_callable__
   ConstOffsetsView
   getOffsets() const;

   /**
    * \brief Iterate over all elements of given segments in parallel and call
    * given lambda function.
    *
    * \tparam Function is a type of the lambda function to be performed on each element.
    * \param begin defines begining of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param end defines end of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param function is the lambda function to be applied on the elements of the segments.
    *
    * Declaration of the lambda function \e function is supposed to be
    *
    * ```
    * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {...}
    * ```
    * where \e segmentIdx is index of segment where given element belong to,
    * \e localIdx is rank of the element within the segment and \e globalIdx is
    * index of the element within the related container.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_forElements.cpp
    * \par Output
    * \include SegmentsExample_CSR_forElements.out
    */
   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& function ) const;

   /**
    * \brief Call \ref TNL::Algorithms::Segments::CSR::forElements for all
    * elements of the segments.
    *
    * See \ref TNL::Algorithms::Segments::CSR::forElements for more details.
    */
   template< typename Function >
   void
   forAllElements( Function&& function ) const;

   /**
    * \brief Iterate over all segments in parallel and call given lambda function.
    *
    * \tparam Function is a type of the lambda function to be performed on each segment.
    * \param begin defines begining of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param end defines end of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param function is the lambda function to be applied on the elements of the segments.
    *
    *  Declaration of the lambda function \e function is supposed to be
    *
    * ```
    * auto f = [=] __cuda_callable__ ( const SegmentView& segment ) {...}
    * ```
    * where \e segment represents given segment (see \ref TNL::Algorithms::Segments::SegmentView).
    * Its type is given by \ref SegmentViewType.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_forSegments.cpp
    * \par Output
    * \include SegmentsExample_CSR_forSegments.out
    */
   template< typename Function >
   void
   forSegments( IndexType begin, IndexType end, Function&& function ) const;

   /**
    * \brief Call \ref TNL::Algorithms::Segments::CSR::forSegments for all segments.
    *
    * See \ref TNL::Algorithms::Segments::CSR::forSegments for more details.
    */
   template< typename Function >
   void
   forAllSegments( Function&& function ) const;

   /**
    * \brief Call \ref TNL::Algorithms::Segments::CSR::forSegments sequentially
    * for particular segments.
    *
    * With this method, the given segments are processed sequentially
    * one-by-one. This is usefull for example for printing of segments based
    * data structures or for debugging reasons.
    *
    * \param begin defines begining of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param end defines end of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param function is the lambda function to be applied on the elements of the segments.
    *
    * See \ref TNL::Algorithms::Segments::CSR::forSegments for more details.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_sequentialForSegments.cpp
    * \par Output
    * \include SegmentsExample_CSR_sequentialForSegments.out
    */
   template< typename Function >
   void
   sequentialForSegments( IndexType begin, IndexType end, Function&& function ) const;

   /**
    * \brief Call \ref TNL::Algorithms::Segments::CSR::sequentialForSegments for all segments.
    *
    * See \ref TNL::Algorithms::Segments::CSR::sequentialForSegments for more details.
    */
   template< typename Function >
   void
   sequentialForAllSegments( Function&& function ) const;

protected:
   //! \brief Vector view with row offsets used in the CSR format.
   OffsetsView offsets;

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
   bind( OffsetsView offsets );
};

/**
 * \brief Insertion operator of CSR segments to output stream.
 *
 * \tparam Device is the device type of the source segments.
 * \tparam Index is the index type of the source segments.
 * \tparam IndexAllocator is the index allocator of the source segments.
 * \param str is the output stream.
 * \param segments are the source segments.
 * \return reference to the output stream.
 */
template< typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str, const CSRBase< Device, Index >& segments )
{
   return printSegments( str, segments );
}

}  // namespace TNL::Algorithms::Segments

#include "CSRBase.hpp"
