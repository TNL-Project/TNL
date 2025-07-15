// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <initializer_list>
#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "SlicedEllpackView.h"
#include "SortedSegments.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for Sliced Ellpack segments.
 *
 * Sliced Ellpack segments are inspired by the following papers:
 *
 * [1] [T. Oberhuber, A. Suzuki, J. Vacata, *New Row-grouped CSR format for storing sparse matrices on GPU with
 * implementation in CUDA*, Acta Technica, 2011, vol. 56, no. 4, pp. 447-466](https://arxiv.org/abs/1012.2270)
 *
 * [2] [A. Monakov, A. Lokhmotov, A. Avetisyan, *Automatically tuning sparse matrix-vector multiplication
 * for GPU architectures*. In International Conference on High-Performance Embedded Architectures and Compilers,
 * pp. 111-125, 2010.](https://link.springer.com/chapter/10.1007/978-3-642-11515-8_10)
 *
 * This format is a modification of \ref TNL::Algorithms::Segments::Ellpack, where segments are divided into slices.
 * Each slice contains segments of the same size, but segments in different slices can have varying sizes.
 * As a result, this format is more flexible than Ellpack while remaining simpler than CSR.
 * If working with segments that have slight variations in size, Sliced Ellpack can be a suitable choice.
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Organization The organization of elements in the segments—either row-major or column-major order.
 * \tparam SliceSize The size of each slice.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int SliceSize = 32 >
class SlicedEllpack : public SlicedEllpackBase< Device, Index, Organization, SliceSize >
{
   using Base = SlicedEllpackBase< Device, Index, Organization, SliceSize >;

public:
   //! \brief Type of allocator for indices.
   using IndexAllocatorType = IndexAllocator;

   //! \brief Type of segments view.
   using ViewType = SlicedEllpackView< Device, Index, Organization, SliceSize >;

   //! \brief Type of constant segments view.
   using ConstViewType = SlicedEllpackView< Device, std::add_const_t< Index >, Organization, SliceSize >;

   //! \brief Type of container storing offsets of particular segments.
   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   template< typename Device_ = Device,
             typename Index_ = Index,
             typename IndexAllocator_ = typename Allocators::Default< Device_ >::template Allocator< Index_ >,
             ElementsOrganization Organization_ = Organization,
             int SliceSize_ = SliceSize >
   using Self = SlicedEllpack< Device_, Index_, IndexAllocator_, Organization_, SliceSize_ >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = SlicedEllpackView< Device_, Index_, Organization, SliceSize >;

   //! \brief Constructor with no parameters to create empty segments.
   SlicedEllpack() = default;

   //! \brief Copy constructor (makes deep copy).
   SlicedEllpack( const SlicedEllpack& );

   //! \brief Move constructor.
   SlicedEllpack( SlicedEllpack&& ) noexcept = default;

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
   explicit SlicedEllpack( const SizesContainer& segmentsSizes );

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
   SlicedEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   //! \brief Copy-assignment operator (makes a deep copy).
   SlicedEllpack&
   operator=( const SlicedEllpack& segments );

   //! \brief Move-assignment operator.
   SlicedEllpack&
   operator=( SlicedEllpack&& ) noexcept( false );

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
   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
   SlicedEllpack&
   operator=( const SlicedEllpack< Device_, Index_, IndexAllocator_, Organization_, SliceSize >& segments );

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
    * \param sizes is an instance of the container with segments sizes.
    */
   template< typename SizesHolder = OffsetsContainer >
   void
   setSegmentsSizes( const SizesHolder& sizes );

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
   OffsetsContainer sliceOffsets;
   OffsetsContainer sliceSegmentSizes;
};

/**
 * \brief Alias for row-major SlicedEllpack segments.
 *
 * See \ref TNL::Algorithms::Segments::SlicedEllpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int SliceSize = 32 >
using RowMajorSlicedEllpack = SlicedEllpack< Device, Index, IndexAllocator, RowMajorOrder, SliceSize >;

/**
 * \brief Alias for column-major SlicedEllpack segments.
 *
 * See \ref TNL::Algorithms::Segments::SlicedEllpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int SliceSize = 32 >
using ColumnMajorSlicedEllpack = SlicedEllpack< Device, Index, IndexAllocator, ColumnMajorOrder, SliceSize >;

/**
 * \brief Alias for sorted SlicedEllpack segments.
 *
 * See \ref TNL::Algorithms::Segments::SlicedEllpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization =
             TNL::Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int SliceSize = 32 >
using SortedSlicedEllpack = SortedSegments< SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize > >;

/**
 * \brief Alias sorted for row-major SlicedEllpack segments.
 *
 * See \ref TNL::Algorithms::Segments::SlicedEllpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int SliceSize = 32 >
using SortedRowMajorSlicedEllpack = SortedSegments< RowMajorSlicedEllpack< Device, Index, IndexAllocator, SliceSize > >;

/**
 * \brief Alias for sorted column-major SlicedEllpack segments.
 *
 * See \ref TNL::Algorithms::Segments::SlicedEllpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int SliceSize = 32 >
using SortedColumnMajorSlicedEllpack = SortedSegments< ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, SliceSize > >;

template< typename Segments >
struct isSlicedEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int SliceSize >
struct isSlicedEllpackSegments< SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize > > : std::true_type
{};

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
struct isSlicedEllpackSegments< SlicedEllpackView< Device, Index, Organization, SliceSize > > : std::true_type
{};

//! \brief Returns true if the given type is SlicedEllpack segments.
template< typename Segments >
inline constexpr bool isSlicedEllpackSegments_v = isSlicedEllpackSegments< Segments >::value;

template< typename Segments >
struct isSortedSlicedEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int SliceSize >
struct isSortedSlicedEllpackSegments< SortedSlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize > >
: std::true_type
{};

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
struct isSortedSlicedEllpackSegments< SortedSlicedEllpackView< Device, Index, Organization, SliceSize > > : std::true_type
{};

//! \brief Returns true if the given type is sorted Ellpack segments.
template< typename Segments >
inline constexpr bool isSortedSlicedEllpackSegments_v = isSortedSlicedEllpackSegments< Segments >::value;

}  // namespace TNL::Algorithms::Segments

#include "SlicedEllpack.hpp"
