// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "CSR.h"
#include "AdaptiveCSRView.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for Adaptive CSR segments format.
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

   /**
    * \brief Constructor with no parameters to create empty segments.
    */
   AdaptiveCSR() = default;

   /**
    * \brief Copy constructor (makes deep copy).
    */
   AdaptiveCSR( const AdaptiveCSR& segments );

   /**
    * \brief Move constructor.
    */
   AdaptiveCSR( AdaptiveCSR&& ) noexcept = default;

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
   AdaptiveCSR( const SizesContainer& segmentsSizes );

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
   AdaptiveCSR( const std::initializer_list< ListIndex >& segmentsSizes );

   //! \brief Copy-assignment operator (makes a deep copy).
   AdaptiveCSR&
   operator=( const AdaptiveCSR& segments );

   //! \brief Move-assignment operator.
   AdaptiveCSR&
   operator=( AdaptiveCSR&& ) noexcept( false );

   /**
    * \brief Assignment operator with CSR segments with different template parameters.
    *
    * It makes a deep copy of the source segments.
    *
    * \tparam Device_ is device type of the source segments.
    * \tparam Index_ is the index type of the source segments.
    * \tparam IndexAllocator_ is the index allocator of the source segments.
    * \param segments is the source segments object.
    * \return reference to this instance.
    */
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   AdaptiveCSR&
   operator=( const AdaptiveCSR< Device_, Index_, IndexAllocator_ >& segments );

   /**
    * \brief Returns a view for this instance of CSR segments which can by used
    * for example in lambda functions running in GPU kernels.
    */
   [[nodiscard]] ViewType
   getView();

   /**
    * \brief Returns a constant view for this instance of CSR segments which
    * can by used for example in lambda functions running in GPU kernels.
    */
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

   /**
    * \brief Reset the segments to empty states.
    *
    * It means that there is no segment in the CSR segments.
    */
   void
   reset();

   [[nodiscard]]
   const BlocksView*
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

}  // namespace TNL::Algorithms::Segments

#include "AdaptiveCSR.hpp"