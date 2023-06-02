// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/SegmentView.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for CSR segments format.
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam IndexAllocator is allocator for supporting index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class CSR
{
public:
   /**
    * \brief The device where the segments are operating.
    */
   using DeviceType = Device;

   /**
    * \brief The type used for indexing of segments elements.
    */
   using IndexType = std::remove_const_t< Index >;

   /**
    * \brief Type of container storing offsets of particular rows.
    */
   using OffsetsContainer = Containers::Vector< Index, DeviceType, IndexType, IndexAllocator >;

   using ConstOffsetsView = typename OffsetsContainer::ConstViewType;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = CSRView< Device_, Index_ >;

   /**
    * \brief Type of segments view.1
    */
   using ViewType = CSRView< Device, Index >;

   /**
    * \brief Type of constant segments view.
    */
   using ConstViewType = CSRView< Device, std::add_const_t< IndexType > >;

   /**
    * \brief Accessor type fro one particular segment.
    */
   using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;

   /**
    * \brief This functions says that CSR format is always organised in
    * row-major order.
    */
   [[nodiscard]] static constexpr ElementsOrganization
   getOrganization()
   {
      return RowMajorOrder;
   }

   /**
    * \brief This function says that CSR format does not use padding elements.
    */
   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return false;
   }

   /**
    * \brief Construct with no parameters to create empty segments.
    */
   CSR() = default;

   /**
    * \brief Construct with segments sizes.
    *
    * The number of segments is given by the size of \e segmentsSizes.
    * Particular elements of this container define sizes of particular
    * segments.
    *
    * \tparam SizesContainer is a type of container for segments sizes.  It can
    *    be \ref TNL::Containers::Array or \ref TNL::Containers::Vector for
    *    example.
    * \param segmentsSizes is an instance of the container with the segments sizes.
    *
    * See the following example:
    *
    * \includelineno Algorithms/Segments/SegmentsExample_CSR_constructor_1.cpp
    *
    * The result looks as follows:
    *
    * \include SegmentsExample_CSR_constructor_1.out
    */
   template< typename SizesContainer >
   CSR( const SizesContainer& segmentsSizes );

   /**
    * \brief Construct with segments sizes in initializer list..
    *
    * The number of segments is given by the size of \e segmentsSizes.
    * Particular elements of this initializer list define sizes of particular
    * segments.
    *
    * \tparam ListIndex is a type of indexes of the initializer list.
    * \param segmentsSizes is an instance of the container with the segments sizes.
    *
    * See the following example:
    *
    * \includelineno Algorithms/Segments/SegmentsExample_CSR_constructor_2.cpp
    *
    * The result looks as follows:
    *
    * \include SegmentsExample_CSR_constructor_2.out
    */
   template< typename ListIndex >
   CSR( const std::initializer_list< ListIndex >& segmentsSizes );

   /**
    * \brief Copy constructor.
    *
    * \param segments are the source segments.
    */
   CSR( const CSR& segments ) = default;

   /**
    * \brief Move constructor.
    *
    * \param segments  are the source segments.
    */
   CSR( CSR&& segments ) noexcept = default;

   /**
    * \brief Returns string with serialization type.
    *
    * \return String with the serialization type.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_getSerializationType.cpp
    * \par Output
    * \include SegmentsExample_CSR_getSerializationType.out
    */
   [[nodiscard]] static std::string
   getSerializationType();

   /**
    * \brief Returns string with segments type.
    *
    * \return \ref String with the segments type.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_getSegmentsType.cpp
    * \par Output
    * \include SegmentsExample_CSR_getSegmentsType.out
    */
   [[nodiscard]] static String
   getSegmentsType();

   /**
    * \brief Set sizes of particular segments.
    *
    * \tparam SizesContainer is a container with segments sizes. It can be
    * \ref TNL::Containers::Array or \ref TNL::Containers::Vector for example.
    *
    * \param segmentsSizes is an instance of the container with segments sizes.
    */
   template< typename SizesContainer >
   void
   setSegmentsSizes( const SizesContainer& segmentsSizes );

   /**
    * \brief Reset the segments to empty states.
    *
    * It means that there is no segment in the CSR segments.
    */
   void
   reset();

   /**
    * \brief Getter of a view object.
    *
    * \return View for this instance of CSR segments which can by used for
    * example in lambda functions running in GPU kernels.
    */
   [[nodiscard]] ViewType
   getView();

   /**
    * \brief Getter of a view object for constants instances.
    *
    * \return View for this instance of CSR segments which can by used for
    * example in lambda functions running in GPU kernels.
    */
   [[nodiscard]] ConstViewType
   getConstView() const;

   /**
    * \brief Getter of number of segments.
    *
    * \return number of segments within this object.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentsCount() const;

   /**
    * \brief Returns size of particular segment.
    *
    * \return size of the segment number \e segmentIdx.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   /***
    * \brief Returns number of elements managed by all segments.
    *
    * \return number of elements managed by all segments.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const;

   /**
    * \brief Returns number of elements that needs to be allocated by a
    * container connected to this segments.
    *
    * \return size of container connected to this segments.
    */
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

   /**
    * \brief Returns reference on constant vector with row offsets used in the CSR format.
    *
    * \return reference on constant vector with row offsets used in the CSR format.
    */
   [[nodiscard]] const OffsetsContainer&
   getOffsets() const;

   /**
    * \brief Returns reference on vector with row offsets used in the CSR format.
    *
    * \return reference on vector with row offsets used in the CSR format.
    */
   [[nodiscard]] OffsetsContainer&
   getOffsets();

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
   sequentialForAllSegments( Function&& f ) const;

   /**
    * \brief Assignment operator.
    *
    * It makes a deep copy of the source segments.
    *
    * \param source are the CSR segments to be assigned.
    * \return reference to this instance.
    */
   CSR&
   operator=( const CSR& source ) = default;

   /**
    * \brief Assignment operator with CSR segments with different template parameters.
    *
    * It makes a deep copy of the source segments.
    *
    * \tparam Device_ is device type of the source segments.
    * \tparam Index_ is the index type of the source segments.
    * \tparam IndexAllocator_ is the index allocator of the source segments.
    * \param source is the source segments object.
    * \return reference to this instance.
    */
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   CSR&
   operator=( const CSR< Device_, Index_, IndexAllocator_ >& source );

   /**
    * \brief Method for saving the segments to a file in a binary form.
    *
    * \param file is the target file.
    */
   void
   save( File& file ) const;

   /**
    * \brief Method for loading the segments from a file in a binary form.
    *
    * \param file is the source file.
    */
   void
   load( File& file );

protected:
   OffsetsContainer offsets;
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
template< typename Device, typename Index, typename IndexAllocator >
std::ostream&
operator<<( std::ostream& str, const CSR< Device, Index, IndexAllocator >& segments )
{
   return printSegments( str, segments );
}

}  // namespace TNL::Algorithms::Segments

#include <TNL/Algorithms/Segments/CSR.hpp>
