// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "CSR.h"
#include "AdaptiveCSRView.h"
#include "SortedSegments.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for the Adaptive CSR segments format.
 *
 * Adaptive CSR segments are inspired by the following paper:
 *
 * [M. Daga and J. L. Greathouse, *Structural Agnostic SpMV: Adapting CSR-Adaptive for Irregular Matrices*,
 * 2015 IEEE 22nd International Conference on High Performance Computing (HiPC), Bengaluru, India, 2015, pp.
 * 64-74.](https://ieeexplore.ieee.org/document/7397620).
 *
 * This format extends the [Compressed Sparse Row (CSR) format](https://en.wikipedia.org/wiki/Sparse_matrix).
 * The data are organized similarly to CSR, but the segments contain additional meta-information
 * to optimize GPU thread mapping. In particular, reduction operations can benefit from this extension.
 * However, reading the meta-information introduces overhead, which may impact performance,
 * especially for small segments. Use this segment type only if you anticipate significant variation
 * in segment sizes.
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing the elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class AdaptiveCSR : public CSR< Device, Index, IndexAllocator >
{
   using Base = CSR< Device, Index, IndexAllocator >;

public:
   //! \brief Type of segments view.
   using ViewType = AdaptiveCSRView< Device, Index >;

   //! \brief Type of constant segments view.
   using ConstViewType = AdaptiveCSRView< Device, std::add_const_t< Index > >;

   using BlocksType = typename ViewType::BlocksType;
   using BlocksView = typename BlocksType::ViewType;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = AdaptiveCSRView< Device_, Index_ >;

   /**
    * \brief Templated type for creating AdaptiveCSR segments with different template parameters.
    *
    * \tparam Device_ is alternative device type.
    * \tparam Index_ is alternative index type.
    * \tparam IndexAllocator_ is alternative index allocator type.
    */
   template< typename Device_ = Device,
             typename Index_ = Index,
             typename IndexAllocator_ = typename Allocators::Default< Device_ >::template Allocator< Index_ > >
   using Self = AdaptiveCSR< Device_, Index_, IndexAllocator_ >;

   //! \brief Constructor with no parameters to create empty segments.
   AdaptiveCSR();

   //! \brief Copy constructor (makes deep copy).
   AdaptiveCSR( const AdaptiveCSR& segments );

   //! \brief Move constructor.
   AdaptiveCSR( AdaptiveCSR&& ) noexcept = default;

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
   AdaptiveCSR( const SizesContainer& segmentsSizes );

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
   AdaptiveCSR( const std::initializer_list< ListIndex >& segmentsSizes );

   //! \brief Copy-assignment operator (makes a deep copy).
   AdaptiveCSR&
   operator=( const AdaptiveCSR& segments );

   //! \brief Move-assignment operator.
   AdaptiveCSR&
   operator=( AdaptiveCSR&& ) noexcept( false );

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
   AdaptiveCSR&
   operator=( const AdaptiveCSR< Device_, Index_, IndexAllocator_ >& segments );

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

   //! \brief Returns a view with blocks used in the Adaptive CSR format.
   [[nodiscard]] const BlocksView*
   getBlocks() const;

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
   [[nodiscard]] static constexpr int
   MaxValueSizeLog()
   {
      return ViewType::MaxValueSizeLog();
   }

   [[nodiscard]] static int
   getSizeValueLog( const int& i )
   {
      return detail::CSRAdaptiveKernelParameters<>::getSizeValueLog( i );
   }

   template< int SizeOfValue, typename Offsets >
   Index
   findLimit( Index start, const Offsets& offsets, Index size, detail::Type& type );

   template< int SizeOfValue, typename Offsets >
   void
   initValueSize( const Offsets& offsets );

   /**
    * \brief  blocksArray[ i ] stores blocks for sizeof( Value ) == 2^i.
    */
   BlocksType blocksArray[ MaxValueSizeLog() ];

   ViewType view;
};

/**
 * \brief Alias for sorted segments based on AdaptiveCSR segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using SortedAdaptiveCSR = SortedSegments< AdaptiveCSR< Device, Index, IndexAllocator > >;

template< typename Segments >
struct isAdaptiveCSRSegments : std::false_type
{};

template< typename Device, typename Index, typename IndexAllocator >
struct isAdaptiveCSRSegments< AdaptiveCSR< Device, Index, IndexAllocator > > : std::true_type
{};

template< typename Device, typename Index >
struct isAdaptiveCSRSegments< AdaptiveCSRView< Device, Index > > : std::true_type
{};

/**
 * \brief Returns true if the given type is AdaptiveCSR segments.
 *
 * \tparam Segments The type of the segments.
 */
template< typename Segments >
inline constexpr bool isAdaptiveCSRSegments_v = isAdaptiveCSRSegments< Segments >::value;

template< typename Device, typename Index, typename IndexAllocator >
struct isSortedSegments< SortedAdaptiveCSR< Device, Index, IndexAllocator > > : std::true_type
{};

template< typename Device, typename Index >
struct isSortedSegments< SortedAdaptiveCSRView< Device, Index > > : std::true_type
{};

}  // namespace TNL::Algorithms::Segments

#include "AdaptiveCSR.hpp"
