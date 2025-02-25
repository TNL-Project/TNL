// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "ChunkedEllpackView.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for Chunked Ellpack segments.
 *
 * Chunked Ellpack segments are inspired by the paper
 * [Heller M., Oberhuber T., Improved Row-grouped CSR Format for Storing of Sparse Matrices on GPU, Proceedings of Algoritmy
 * 2012, 2012, Handlovičová A., Minarechová Z. and Ševčovič D. (ed.), pages
 * 282-290](https://geraldine.fjfi.cvut.cz/~oberhuber/data/vyzkum/publikace/12-heller-oberhuber-improved-rgcsr-format.pdf).
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Organization The organization of the elements in the segments—either row-major or column-major order.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class ChunkedEllpack : public ChunkedEllpackBase< Device, Index, Organization >
{
   using Base = ChunkedEllpackBase< Device, Index, Organization >;

public:
   //! \brief Type of segments view.
   using ViewType = ChunkedEllpackView< Device, Index, Organization >;

   //! \brief Type of constant segments view.
   using ConstViewType = typename ViewType::ConstViewType;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = ChunkedEllpackView< Device_, Index_, Organization >;

   //! \brief Type of container storing offsets of particular segments.
   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   using SliceInfoAllocator = typename Allocators::Default< Device >::template Allocator< typename Base::SliceInfoType >;
   using SliceInfoContainer =
      Containers::Array< typename TNL::copy_const< typename Base::SliceInfoType >::template from< Index >::type,
                         Device,
                         Index,
                         SliceInfoAllocator >;

   //! \brief Constructor with no parameters to create empty segments.
   ChunkedEllpack() = default;

   //! \brief Copy constructor (makes deep copy).
   ChunkedEllpack( const ChunkedEllpack& segments );

   //! \brief Move constructor.
   ChunkedEllpack( ChunkedEllpack&& segments ) noexcept = default;

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
   ChunkedEllpack( const SizesContainer& segmentsSizes );

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
   ChunkedEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   //! \brief Copy-assignment operator (makes a deep copy).
   ChunkedEllpack&
   operator=( const ChunkedEllpack& segments );

   //! \brief Move-assignment operator.
   ChunkedEllpack&
   operator=( ChunkedEllpack&& ) noexcept( false );

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
   ChunkedEllpack&
   operator=( const ChunkedEllpack< Device_, Index_, IndexAllocator_, Organization_ >& segments );

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
   template< typename SizesContainer >
   void
   resolveSliceSizes( SizesContainer& segmentsSizes );

   template< typename SizesContainer >
   bool
   setSlice( SizesContainer& segmentsSizes, Index sliceIndex, Index& elementsToAllocation );

   //! \brief For each segment, this keeps index of the first chunk within a slice.
   OffsetsContainer segmentToChunkMapping;

   //! \brief For each segment, this keeps index of the slice which contains the segment.
   OffsetsContainer segmentToSliceMapping;

   OffsetsContainer chunksToSegmentsMapping;

   //! \brief Keeps index of the first segment index.
   OffsetsContainer segmentPointers;

   SliceInfoContainer slices;
};

/**
 * \brief Data structure for row-major Chunked Ellpack segments.
 *
 * See \ref TNL::Algorithms::Segments::ChunkedEllpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
struct RowMajorChunkedEllpack : public ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >
{
   using BaseType = ChunkedEllpack< Device, Index, IndexAllocator, RowMajorOrder >;

   //! \brief Constructor with no parameters to create empty segments.
   RowMajorChunkedEllpack() = default;

   //! \brief Copy constructor (makes deep copy).
   RowMajorChunkedEllpack( const RowMajorChunkedEllpack& );

   //! \brief Move constructor.
   RowMajorChunkedEllpack( RowMajorChunkedEllpack&& ) noexcept = default;

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
   explicit RowMajorChunkedEllpack( const SizesContainer& segmentsSizes )
   : BaseType( segmentsSizes )
   {}

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
   RowMajorChunkedEllpack( const std::initializer_list< ListIndex >& segmentsSizes )
   : BaseType( segmentsSizes )
   {}
};

/**
 * \brief Data structure for column-major Chunked Ellpack segments.
 *
 * See \ref TNL::Algorithms::Segments::ChunkedEllpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
struct ColumnMajorChunkedEllpack : public ChunkedEllpack< Device, Index, IndexAllocator, ColumnMajorOrder >
{
   using BaseType = ChunkedEllpack< Device, Index, IndexAllocator, ColumnMajorOrder >;

   //! \brief Constructor with no parameters to create empty segments.
   ColumnMajorChunkedEllpack() = default;

   //! \brief Copy constructor (makes deep copy).
   ColumnMajorChunkedEllpack( const ColumnMajorChunkedEllpack& );

   //! \brief Move constructor.
   ColumnMajorChunkedEllpack( ColumnMajorChunkedEllpack&& ) noexcept = default;

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
   explicit ColumnMajorChunkedEllpack( const SizesContainer& segmentsSizes )
   : BaseType( segmentsSizes )
   {}

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
   ColumnMajorChunkedEllpack( const std::initializer_list< ListIndex >& segmentsSizes )
   : BaseType( segmentsSizes )
   {}
};

template< typename Segments >
struct isChunkedEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
struct isChunkedEllpackSegments< ChunkedEllpack< Device, Index, IndexAllocator, Organization > > : std::true_type
{};

template< typename Device, typename Index, ElementsOrganization Organization >
struct isChunkedEllpackSegments< ChunkedEllpackView< Device, Index, Organization > > : std::true_type
{};

//! \brief Returns true if the given type is Chunked Ellpack segments.
template< typename Segments >
inline constexpr bool isChunkedEllpackSegments_v = isChunkedEllpackSegments< Segments >::value;

}  // namespace TNL::Algorithms::Segments

#include "ChunkedEllpack.hpp"
