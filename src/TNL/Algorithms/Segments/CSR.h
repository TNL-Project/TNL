// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <initializer_list>

#include <TNL/Containers/Vector.h>
#include <TNL/TypeTraits.h>
#include <TNL/Allocators/Default.h>

#include "CSRView.h"
#include "SortedSegments.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for CSR segments.
 *
 * CSR segments are inspired by the [Compressed Sparse Row (CSR) format](https://en.wikipedia.org/wiki/Sparse_matrix),
 * which is widely used for storing sparse matrices. It is the most
 * popular format due to its versatility, making it the preferred choice
 * for segment representation.
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class CSR : public CSRBase< Device, Index >
{
   using Base = CSRBase< Device, Index >;

public:
   //! \brief Type of segments view.
   using ViewType = CSRView< Device, Index >;

   //! \brief Type of constant segments view.
   using ConstViewType = CSRView< Device, std::add_const_t< Index > >;

   /**
    * \brief Templated type for creating CSR segments with different template parameters.
    *
    * \tparam Device_ is alternative device type.
    * \tparam Index_ is alternative index type.
    * \tparam IndexAllocator_ is alternative index allocator type.
    */
   template< typename Device_ = Device,
             typename Index_ = Index,
             typename IndexAllocator_ = typename Allocators::Default< Device_ >::template Allocator< Index_ > >
   using Self = CSR< Device_, Index_, IndexAllocator_ >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = CSRView< Device_, Index_ >;

   //! \brief Type of container storing offsets of particular segments.
   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   using IndexAllocatorType = IndexAllocator;

   //! \brief Constructor with no parameters to create empty segments.
   CSR();

   //! \brief Copy constructor (makes deep copy).
   CSR( const CSR& segments );

   //! \brief Move constructor.
   CSR( CSR&& ) noexcept = default;

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
   template< typename SizesContainer, typename T = std::enable_if_t< IsArrayType< SizesContainer >::value > >
   CSR( const SizesContainer& segmentsSizes );

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
   CSR( const std::initializer_list< ListIndex >& segmentsSizes );

   //! \brief Copy-assignment operator (makes a deep copy).
   CSR&
   operator=( const CSR& segments );

   //! \brief Move-assignment operator.
   CSR&
   operator=( CSR&& ) noexcept( false );

   /**
    * \brief Assignment operator for segments with different template parameters.
    *
    * Performs a deep copy of the source segments.
    *
    * \tparam Device_ The device type of the source segments.
    * \tparam Index_ The index type of the source segments.
    * \tparam IndexAllocator_ The index allocator type of the source segments.
    * \param segments The source segments object.
    * \return A reference to this instance.
    */
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   CSR&
   operator=( const CSR< Device_, Index_, IndexAllocator_ >& segments );

   //! \brief Returns a view for this instance of segments which can by used
   //! for example in lambda functions running in GPU kernels.
   [[nodiscard]] ViewType
   getView();

   //! \brief Returns a constant view for this instance of segments which
   //! can by used for example in lambda functions running in GPU kernels.
   [[nodiscard]] ConstViewType
   getConstView() const;

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
   OffsetsContainer offsets;
};

/**
 * \brief Alias for sorted segments based on CSR segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using SortedCSR = SortedSegments< CSR< Device, Index, IndexAllocator > >;

template< typename Segments >
struct isCSRSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator >
struct isCSRSegments< CSR< Device, Index, IndexAllocator > > : std::true_type
{};

template< typename Device, typename Index >
struct isCSRSegments< CSRView< Device, Index > > : std::true_type
{};

//! \brief Returns true if the given type is CSR segments.
template< typename Segments >
inline constexpr bool isCSRSegments_v = isCSRSegments< Segments >::value;

template< typename Device, typename Index, typename IndexAllocator >
struct isSortedSegments< SortedSegments< CSR< Device, Index, IndexAllocator > > > : std::true_type
{};

template< typename Device, typename Index >
struct isSortedSegments< SortedCSRView< Device, Index > > : std::true_type
{};

}  // namespace TNL::Algorithms::Segments

#include "CSR.hpp"
