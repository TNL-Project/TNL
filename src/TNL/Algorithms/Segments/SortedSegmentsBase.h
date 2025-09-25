// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief \e SortedSegmentsBase serves as a base class for \ref TNL::Algorithms::Segments::SortedSegments
 * and \ref TNL::Algorithms::Segments::SortedSegmentsView.
 *
 * The sorted segments are inspired by SELL-C-sigma sparse matrix storage format:
 *
 * Kreutzer, Moritz, et al. "A unified sparse matrix data format for efficient general
 * sparse matrix-vector multiplication on modern processors with wide SIMD units."
 * SIAM Journal on Scientific Computing 36.5 (2014): C401-C423.
 *
 * \tparam EmbeddedSegments is a type of segments used to manage the data.
 */
template< typename EmbeddedSegmentsView_ >
class SortedSegmentsBase
{
public:
   using EmbeddedSegmentsView = EmbeddedSegmentsView_;

   using EmbeddedSegmentsConstView = typename EmbeddedSegmentsView::ConstViewType;

   //! \brief The device where the segments are operating.
   using DeviceType = typename EmbeddedSegmentsView::DeviceType;

   //! \brief The type used for indexing of segments elements.
   using IndexType = typename EmbeddedSegmentsView::IndexType;

   //! \brief Accessor type for one particular segment.
   using SegmentViewType = typename EmbeddedSegmentsView::SegmentViewType;

   using PermutationView = typename Containers::VectorView< IndexType, DeviceType, IndexType >;

   using ConstPermutationView = typename Containers::VectorView< std::add_const_t< IndexType >, DeviceType, IndexType >;

   //! \brief Returns the data layout (it is always row-major order).
   [[nodiscard]] static constexpr ElementsOrganization
   getOrganization()
   {
      return EmbeddedSegmentsView::getOrganization();
   }

   //! \brief This function denotes if the underlying segments use the padding elements.
   [[nodiscard]] static constexpr bool
   havePadding()
   {
      return EmbeddedSegmentsView::havePadding();
   }

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   SortedSegmentsBase() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   SortedSegmentsBase( const SortedSegmentsBase& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   SortedSegmentsBase( SortedSegmentsBase&& ) noexcept = default;

   //! \brief Copy-assignment operator.
   SortedSegmentsBase&
   operator=( const SortedSegmentsBase& ) = delete;

   //! \brief Move-assignment operator.
   SortedSegmentsBase&
   operator=( SortedSegmentsBase&& ) = delete;

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

   [[nodiscard]] __cuda_callable__
   EmbeddedSegmentsConstView
   getEmbeddedSegmentsView() const;

   [[nodiscard]] __cuda_callable__
   EmbeddedSegmentsView
   getEmbeddedSegmentsView();

   //! \brief Returns a modifiable vector view with segments permutation.
   [[nodiscard]] __cuda_callable__
   PermutationView
   getSegmentsPermutationView();

   //! \brief Returns a constant vector view with segments permutation..
   [[nodiscard]] __cuda_callable__
   ConstPermutationView
   getSegmentsPermutationView() const;

   //! \brief Returns a modifiable vector view with inverse segments permutation.
   [[nodiscard]] __cuda_callable__
   PermutationView
   getInverseSegmentsPermutationView();

   //! \brief Returns a constant vector view with inverse segments permutation..
   [[nodiscard]] __cuda_callable__
   ConstPermutationView
   getInverseSegmentsPermutationView() const;

   //! \brief Sets the value of sigma.
   __cuda_callable__
   void
   setSigma( IndexType value );

   //! \brief Gets the value of sigma.
   [[nodiscard]] __cuda_callable__
   IndexType
   getSigma() const;

protected:
   EmbeddedSegmentsView embeddedSegmentsView;

   //! \brief Vector view with the segments permutation.
   PermutationView segmentsPermutationView, inverseSegmentsPermutationView;

   IndexType sigma = -1;

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
   bind( const EmbeddedSegmentsView& embeddedSegmentsView,
         const PermutationView& segmentsPermutation,
         const PermutationView& inverseSegmentsPermutation );
};

}  // namespace TNL::Algorithms::Segments

#include "SortedSegmentsBase.hpp"
#include "print.h"
