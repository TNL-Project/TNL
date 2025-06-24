// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "BiEllpackView.h"
#include "SortedSegments.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for Bisection Ellpack segments.
 *
 * Bisection Ellpack segments are inspired by the following paper:
 *
 * [C. Zheng, S. Gu, T.-X. Gu, B. Yang, X.-P. Liu,
 * *BiELL: A bisection ELLPACK-based storage format for optimizing SpMV on GPUs*,
 * Journal of Parallel and Distributed Computing, Volume 74, Issue 7, 2014, pp.
 * 2639-2647](https://www.sciencedirect.com/science/article/pii/S0743731514000458).
 *
 * This format is designed to improve load balancing for segments with unevenly distributed sizes.
 * It uses more meta-information compared to, for example, \ref TNL::Algorithms::Segments::CSR,
 * which can introduce overhead. However, it can be beneficial for reduction operations,
 * particularly for longer segments with significant size variations.
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Organization The organization of the elements in the segments—either row-major or column-major order.
 * \tparam WarpSize The warp size used for the segments.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int WarpSize = Backend::getWarpSize() >
class BiEllpack : public BiEllpackBase< Device, Index, Organization, WarpSize >
{
   using Base = BiEllpackBase< Device, Index, Organization, WarpSize >;

public:
   //! \brief Type of segments view.
   using ViewType = BiEllpackView< Device, Index, Organization, WarpSize >;

   //! \brief Type of constant segments view.
   using ConstViewType = typename ViewType::ConstViewType;

   /**
    * \brief Templated type for creating BiEllpack segments with different template parameters.
    *
    * \tparam Device_ is alternative device type.
    * \tparam Index_ is alternative index type.
    * \tparam IndexAllocator_ is alternative index allocator type.
    */
   template< typename Device_ = Device,
             typename Index_ = Index,
             typename IndexAllocator_ = typename Allocators::Default< Device_ >::template Allocator< Index_ >,
             ElementsOrganization Organization_ = Organization,
             int WarpSize_ = WarpSize >
   using Self = BiEllpack< Device_, Index_, IndexAllocator_, Organization_, WarpSize_ >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_ = Device, typename Index_ = Index >
   using ViewTemplate = BiEllpackView< Device_, Index_, Organization, WarpSize >;

   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   using IndexAllocatorType = IndexAllocator;

   //! \brief Constructor with no parameters to create empty segments.
   BiEllpack() = default;

   //! \brief Copy constructor (makes deep copy).
   BiEllpack( const BiEllpack& segments );

   //! \brief Move constructor.
   BiEllpack( BiEllpack&& segments ) noexcept = default;

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
   template< typename SizesContainer >
   BiEllpack( const SizesContainer& segmentsSizes );

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
   BiEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   //! \brief Copy-assignment operator (makes a deep copy).
   BiEllpack&
   operator=( const BiEllpack& segments );

   //! \brief Move-assignment operator.
   BiEllpack&
   operator=( BiEllpack&& segments ) noexcept( false );

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
   BiEllpack&
   operator=( const BiEllpack< Device_, Index_, IndexAllocator_, Organization_, WarpSize >& segments );

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
   template< typename SizesHolder >
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

   // This method is public only because of lambda functions in CUDA
   template< typename SizesHolder >
   void
   initGroupPointers( const SizesHolder& segmentsSizes );

protected:
   OffsetsContainer segmentsPermutation;
   OffsetsContainer groupPointers;

   template< typename SizesHolder >
   void
   initSegmentsPermutation( const SizesHolder& segmentsSize );

   template< typename SizesHolder >
   void
   verifySegmentPerm( const SizesHolder& segmentsSizes );

   template< typename SizesHolder >
   void
   verifySegmentLengths( const SizesHolder& segmentsSizes );

   [[nodiscard]] Index
   getStripLength( Index strip ) const;
};

/**
 * \brief Alias for row-major BiEllpack segments.
 *
 * See \ref TNL::Algorithms::Segments::BiEllpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Alignment The alignment of the number of segments (to optimize data
 * alignment, particularly on GPUs).
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int WarpSize = Backend::getWarpSize() >
using RowMajorBiEllpack = BiEllpack< Device, Index, IndexAllocator, RowMajorOrder, WarpSize >;

/**
 * \brief Alias for column-major BiEllpack segments.
 *
 * See \ref TNL::Algorithms::Segments::BiEllpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Alignment The alignment of the number of segments (to optimize data
 * alignment, particularly on GPUs).
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int WarpSize = Backend::getWarpSize() >
using ColumnMajorBiEllpack = BiEllpack< Device, Index, IndexAllocator, ColumnMajorOrder, WarpSize >;

/**
 * \brief Alias for sorted segments based on BiEllpack segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int WarpSize = Backend::getWarpSize() >
using SortedBiEllpack = SortedSegments< BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize > >;

/**
 * \brief Alias for sorted segments based on row-major BiEllpack segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int WarpSize = Backend::getWarpSize() >
using SortedRowMajorBiEllpack = SortedSegments< RowMajorBiEllpack< Device, Index, IndexAllocator, WarpSize > >;

/**
 * \brief Alias for sorted segments based on column-major BiEllpack segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int WarpSize = Backend::getWarpSize() >
using SortedColumnMajorBiEllpack = SortedSegments< ColumnMajorBiEllpack< Device, Index, IndexAllocator, WarpSize > >;

template< typename Segments >
struct isBiEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize_ >
struct isBiEllpackSegments< BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize_ > > : std::true_type
{};

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize_ >
struct isBiEllpackSegments< BiEllpackView< Device, Index, Organization, WarpSize_ > > : std::true_type
{};

//! \brief Returns true if the given type is BiEllpack segments.
template< typename Segments >
inline constexpr bool isBiEllpackSegments_v = isBiEllpackSegments< Segments >::value;

template< typename Segments >
struct isRowMajorBiEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, int WarpSize_ >
struct isRowMajorBiEllpackSegments< RowMajorBiEllpack< Device, Index, IndexAllocator, WarpSize_ > > : std::true_type
{};

template< typename Device, typename Index, int WarpSize_ >
struct isRowMajorBiEllpackSegments< RowMajorBiEllpackView< Device, Index, WarpSize_ > > : std::true_type
{};

//! \brief Returns true if the given type is row-major BiEllpack segments.
template< typename Segments >
inline constexpr bool isRowMajorBiEllpackSegments_v = isRowMajorBiEllpackSegments< Segments >::value;

template< typename Segments >
struct isColumnMajorBiEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, int WarpSize_ >
struct isColumnMajorBiEllpackSegments< ColumnMajorBiEllpack< Device, Index, IndexAllocator, WarpSize_ > > : std::true_type
{};

template< typename Device, typename Index, int WarpSize_ >
struct isColumnMajorBiEllpackSegments< ColumnMajorBiEllpackView< Device, Index, WarpSize_ > > : std::true_type
{};

//! \brief Returns true if the given type is column-major BiEllpack segments.
template< typename Segments >
inline constexpr bool isColumnMajorBiEllpackSegments_v = isColumnMajorBiEllpackSegments< Segments >::value;

template< typename Segments >
struct isSortedBiEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize_ >
struct isSortedBiEllpackSegments< SortedBiEllpack< Device, Index, IndexAllocator, Organization, WarpSize_ > > : std::true_type
{};

template< typename Device, typename Index, ElementsOrganization Organization, int WarpSize_ >
struct isSortedBiEllpackSegments< SortedBiEllpackView< Device, Index, Organization, WarpSize_ > > : std::true_type
{};

//! \brief Returns true if the given type is sorted BiEllpack segments.
template< typename Segments >
inline constexpr bool isSortedBiEllpackSegments_v = isSortedBiEllpackSegments< Segments >::value;

template< typename Segments >
struct isSortedRowMajorBiEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, int WarpSize_ >
struct isSortedRowMajorBiEllpackSegments< SortedRowMajorBiEllpack< Device, Index, IndexAllocator, WarpSize_ > > : std::true_type
{};

template< typename Device, typename Index, int WarpSize_ >
struct isSortedRowMajorBiEllpackSegments< SortedRowMajorBiEllpackView< Device, Index, WarpSize_ > > : std::true_type
{};

//! \brief Returns true if the given type is sorted row-major BiEllpack segments.
template< typename Segments >
inline constexpr bool isSortedRowMajorBiEllpackSegments_v = isSortedRowMajorBiEllpackSegments< Segments >::value;

template< typename Segments >
struct isSortedColumnMajorBiEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, int WarpSize_ >
struct isSortedColumnMajorBiEllpackSegments< SortedColumnMajorBiEllpack< Device, Index, IndexAllocator, WarpSize_ > >
: std::true_type
{};

template< typename Device, typename Index, int WarpSize_ >
struct isSortedColumnMajorBiEllpackSegments< SortedColumnMajorBiEllpackView< Device, Index, WarpSize_ > > : std::true_type
{};

//! \brief Returns true if the given type is sorted column-major BiEllpack segments.
template< typename Segments >
inline constexpr bool isSortedColumnMajorBiEllpackSegments_v = isSortedColumnMajorBiEllpackSegments< Segments >::value;

}  // namespace TNL::Algorithms::Segments

#include "BiEllpack.hpp"
