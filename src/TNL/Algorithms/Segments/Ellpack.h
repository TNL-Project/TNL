// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "EllpackView.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for Ellpack segments.
 *
 * Ellpack segments are inspired by the [Ellpack format](https://dl.acm.org/doi/abs/10.1145/1654059.1654078),
 * which is commonly used for storing sparse matrices on parallel architectures. Compared to CSR,
 * Ellpack has a fixed number of elements per segment, which can be more efficient for certain operations.
 * Therefore, this format may be preferable to CSR if all segments contain approximately the same number of elements.
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Organization The organization of the elements in the segments—either row-major or column-major order.
 * \tparam Alignment The alignment of the number of segments (to optimize data alignment, particularly on GPUs).
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
class Ellpack : public EllpackBase< Device, Index, Organization, Alignment >
{
   using Base = EllpackBase< Device, Index, Organization, Alignment >;

public:
   //! \brief Type of segments view.
   using ViewType = EllpackView< Device, Index, Organization, Alignment >;

   //! \brief Type of constant segments view.
   using ConstViewType = typename ViewType::ConstViewType;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = EllpackView< Device_, Index_, Organization, Alignment >;

   //! \brief Type of container storing offsets of particular segments.
   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   //! \brief Constructor with no parameters to create empty segments.
   Ellpack() = default;

   //! \brief Copy constructor (makes deep copy).
   Ellpack( const Ellpack& segments ) = default;

   //! \brief Move constructor.
   Ellpack( Ellpack&& segments ) noexcept = default;

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
   Ellpack( const SizesContainer& sizes );

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
   Ellpack( const std::initializer_list< ListIndex >& segmentsSizes );

   //! \brief Constructor that initializes segments based on the number of segments and the size of each segment.
   Ellpack( Index segmentsCount, Index segmentSize );

   //! \brief Copy-assignment operator.
   Ellpack&
   operator=( const Ellpack& segments );

   //! \brief Move-assignment operator.
   Ellpack&
   operator=( Ellpack&& ) noexcept;

   /**
    * \brief Assignment operator for segments with different template parameters.
    *
    * Performs a deep copy of the source segments.
    *
    * \tparam Device_ The device type of the source segments.
    * \tparam Index_ The index type of the source segments.
    * \tparam IndexAllocator_ The index allocator type of the source segments.
    * \tparam Organization_ The organization of the elements in the source segments.
    * \tparam Alignment_ The alignment of the number of segments in the source segments.
    * \param segments The source segments object.
    * \return A reference to this instance.
    */
   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_, int Alignment_ >
   Ellpack&
   operator=( const Ellpack< Device_, Index_, IndexAllocator_, Organization_, Alignment_ >& segments );

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
   setSegmentsSizes( const SizesContainer& sizes );

   /**
    * \brief Set sizes of the segments.
    *
    * \param segmentsCount is the number of segments.
    * \param segmentSize is the size of each segment.
    */
   void
   setSegmentsSizes( Index segmentsCount, Index segmentSize );

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
};

/**
 * \brief Data structure for row-major Ellpack segments.
 *
 * See \ref TNL::Algorithms::Segments::Ellpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Alignment The alignment of the number of segments (to optimize data alignment, particularly on GPUs).
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int Alignment = 32 >
struct RowMajorEllpack : public Ellpack< Device, Index, IndexAllocator, RowMajorOrder, Alignment >
{
   using BaseType = Ellpack< Device, Index, IndexAllocator, RowMajorOrder, Alignment >;

   //! \brief Constructor with no parameters to create empty segments.
   RowMajorEllpack() = default;

   //! \brief Copy constructor (makes deep copy).
   RowMajorEllpack( const RowMajorEllpack& segments ) = default;

   //! \brief Move constructor.
   RowMajorEllpack( RowMajorEllpack&& segments ) noexcept = default;

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
   RowMajorEllpack( const SizesContainer& sizes )
   : BaseType( sizes )
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
   RowMajorEllpack( const std::initializer_list< ListIndex >& segmentsSizes )
   : BaseType( segmentsSizes )
   {}

   //! \brief Constructor that initializes segments based on the number of segments and the size of each segment.
   RowMajorEllpack( Index segmentsCount, Index segmentSize )
   : BaseType( segmentsCount, segmentSize )
   {}
};

/**
 * \brief Data structure for column-major Ellpack segments.
 *
 * See \ref TNL::Algorithms::Segments::Ellpack for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Alignment The alignment of the number of segments (to optimize data alignment, particularly on GPUs).
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          int Alignment = 32 >
struct ColumnMajorEllpack : public Ellpack< Device, Index, IndexAllocator, ColumnMajorOrder, Alignment >
{
   using BaseType = Ellpack< Device, Index, IndexAllocator, ColumnMajorOrder, Alignment >;

   //! \brief Constructor with no parameters to create empty segments.
   ColumnMajorEllpack() = default;

   //! \brief Copy constructor (makes deep copy).
   ColumnMajorEllpack( const ColumnMajorEllpack& segments ) = default;

   //! \brief Move constructor.
   ColumnMajorEllpack( ColumnMajorEllpack&& segments ) noexcept = default;

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
   ColumnMajorEllpack( const SizesContainer& sizes )
   : BaseType( sizes )
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
   ColumnMajorEllpack( const std::initializer_list< ListIndex >& segmentsSizes )
   : BaseType( segmentsSizes )
   {}

   //! \brief Constructor that initializes segments based on the number of segments and the size of each segment.
   ColumnMajorEllpack( Index segmentsCount, Index segmentSize )
   : BaseType( segmentsCount, segmentSize )
   {}
};

template< typename Segments >
struct isEllpackSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
struct isEllpackSegments< Ellpack< Device, Index, IndexAllocator, Organization, Alignment > > : std::true_type
{};

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
struct isEllpackSegments< EllpackView< Device, Index, Organization, Alignment > > : std::true_type
{};

template< typename Device, typename Index, typename IndexAllocator, int Alignment >
struct isEllpackSegments< ColumnMajorEllpack< Device, Index, IndexAllocator, Alignment > > : std::true_type
{};

template< typename Device, typename Index, int Alignment >
struct isEllpackSegments< ColumnMajorEllpackView< Device, Index, Alignment > > : std::true_type
{};

template< typename Device, typename Index, typename IndexAllocator, int Alignment >
struct isEllpackSegments< RowMajorEllpack< Device, Index, IndexAllocator, Alignment > > : std::true_type
{};
template< typename Device, typename Index, int Alignment >
struct isEllpackSegments< RowMajorEllpackView< Device, Index, Alignment > > : std::true_type
{};

//! \brief Returns true if the given type is Ellpack segments.
template< typename Segments >
inline constexpr bool isEllpackSegments_v = isEllpackSegments< Segments >::value;

}  // namespace TNL::Algorithms::Segments

#include "Ellpack.hpp"
