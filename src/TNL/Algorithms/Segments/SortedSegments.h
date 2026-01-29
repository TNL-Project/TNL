// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/TypeTraits.h>

#include "SortedSegmentsView.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for sorted segments.
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam EmbeddedSegments The type of embedded segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename EmbeddedSegments, typename IndexAllocator = typename EmbeddedSegments::IndexAllocatorType >
class SortedSegments : public SortedSegmentsBase< typename EmbeddedSegments::ViewType >
{
   using Base = SortedSegmentsBase< typename EmbeddedSegments::ViewType >;

public:
   using typename Base::DeviceType;

   using typename Base::IndexType;

   using typename Base::EmbeddedSegmentsView;

   using typename Base::EmbeddedSegmentsConstView;

   using typename Base::PermutationView;

   using typename Base::ConstPermutationView;

   using EmbeddedSegmentsType = EmbeddedSegments;

   //! \brief Type of segments view.
   using ViewType = SortedSegmentsView< typename EmbeddedSegments::ViewType >;

   //! \brief Type of constant segments view.
   using ConstViewType = SortedSegmentsView< typename EmbeddedSegments::ConstViewType >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_ = DeviceType, typename Index_ = IndexType >
   using ViewTemplate = SortedSegmentsView< typename EmbeddedSegments::template ViewTemplate< Device_, Index_ > >;

   using IndexAllocatorType = IndexAllocator;

   //! \brief Type of container storing offsets of particular segments.
   using PermutationContainer = Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocatorType >;

   //! \brief Constructor with no parameters to create empty segments.
   SortedSegments();

   //! \brief Copy constructor (makes deep copy).
   SortedSegments( const SortedSegments& segments );

   //! \brief Move constructor.
   SortedSegments( SortedSegments&& ) noexcept = default;

   /**
    * \brief Constructor that initializes segments based on their sizes.
    *
    * The number of segments is determined by the size of \e segmentsSizes.
    * Each element in this container specifies the size of a corresponding segment.
    *
    * \tparam SizesContainer The type of container used to store segment sizes.
    *    It can be, for example, \ref TNL::Containers::Array or \ref TNL::Containers::Vector.
    * \param segmentsSizes An instance of the container holding the sizes of the segments.
    *
    * See the following example:
    *
    * \includelineno Algorithms/Segments/SegmentsExample_constructor_1.cpp
    *
    * The expected output is:
    *
    * \include SegmentsExample_constructor_1.out
    */
   template< typename SizesContainer, std::enable_if_t< IsArrayType< SizesContainer >::value, bool > = true >
   SortedSegments( const SizesContainer& segmentsSizes );

   /**
    * \brief Constructor that initializes segments using an initializer list.
    *
    * The number of segments is determined by the size of \e segmentsSizes.
    * Each element in this initializer list specifies the size of a corresponding segment.
    *
    * \tparam ListIndex The type used for indexing elements in the initializer list.
    * \param segmentsSizes An initializer list defining the sizes of the segments.
    *
    * See the following example:
    *
    * \includelineno Algorithms/Segments/SegmentsExample_constructor_2.cpp
    *
    * The expected output is:
    *
    * \include SegmentsExample_constructor_2.out
    */
   template< typename ListIndex >
   SortedSegments( const std::initializer_list< ListIndex >& segmentsSizes );

   //! \brief Copy-assignment operator (makes a deep copy).
   SortedSegments&
   operator=( const SortedSegments& segments );

   //! \brief Move-assignment operator.
   SortedSegments&
   operator=( SortedSegments&& ) noexcept( false );

   //! \brief Copy-assignment operator for segments with different template parameters.
   template< typename EmbeddedSegments_, typename IndexAllocator_ >
   SortedSegments&
   operator=( const SortedSegments< EmbeddedSegments_, IndexAllocator_ >& segments ) noexcept( false );

   //! \brief Move-assignment operator for segments with different template parameters.
   template< typename EmbeddedSegments_, typename IndexAllocator_ >
   SortedSegments&
   operator=( SortedSegments< EmbeddedSegments_, IndexAllocator_ >&& segments ) noexcept( false );

   //! \brief Returns a view for this instance of segments which can by used
   //! for example in lambda functions running in GPU kernels.
   [[nodiscard]] ViewType
   getView();

   //! \brief Returns a constant view for this instance of segments which
   //! can by used for example in lambda functions running in GPU kernels.
   [[nodiscard]] ConstViewType
   getConstView() const;

   //! \brief Returns a reference on embedded segments.
   [[nodiscard]] const EmbeddedSegments&
   getEmbeddedSegments() const;

   //! \brief Returns a reference on segments permutation.
   [[nodiscard]] const PermutationContainer&
   getSegmentsPermutation() const;

   //! \brief Returns a reference on inverse segments permutation.
   [[nodiscard]] const PermutationContainer&
   getInverseSegmentsPermutation() const;

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

   //! \brief Reset the segments to empty states (it means that there is no segment in the segments).
   void
   reset();

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
   EmbeddedSegments embeddedSegments;
   PermutationContainer segmentsPermutation, inverseSegmentsPermutation;
};

template< typename Segments >
struct isSortedSegments : std::false_type
{};

template< typename EmbeddedSegments >
struct isSortedSegments< SortedSegments< EmbeddedSegments > > : std::true_type
{};

template< typename EmbeddedSegments >
struct isSortedSegments< SortedSegmentsView< EmbeddedSegments > > : std::true_type
{};

/**
 * \brief Returns true if the given type is CSR segments.
 *
 * \tparam Segments The type of the segments.
 */
template< typename Segments >
inline constexpr bool isSortedSegments_v = isSortedSegments< Segments >::value;

}  // namespace TNL::Algorithms::Segments

#include "SortedSegments.hpp"
