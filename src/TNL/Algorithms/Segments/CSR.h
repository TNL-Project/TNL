// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "CSRView.h"

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
class CSR : public CSRBase< Device, Index >
{
   using Base = CSRBase< Device, Index >;

public:
   //! \brief Type of segments view.
   using ViewType = CSRView< Device, Index >;

   //! \brief Type of constant segments view.
   using ConstViewType = CSRView< Device, std::add_const_t< Index > >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = CSRView< Device_, Index_ >;

   /**
    * \brief Type of container storing offsets of particular rows.
    */
   using OffsetsContainer = Containers::Vector< Index, Device, typename Base::IndexType, IndexAllocator >;

   /**
    * \brief Constructor with no parameters to create empty segments.
    */
   CSR() = default;

   /**
    * \brief Copy constructor (makes deep copy).
    */
   CSR( const CSR& segments );

   /**
    * \brief Move constructor.
    */
   CSR( CSR&& ) noexcept = default;

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

   //! \brief Copy-assignment operator (makes a deep copy).
   CSR&
   operator=( const CSR& segments );

   //! \brief Move-assignment operator.
   CSR&
   operator=( CSR&& ) noexcept( false );

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
   CSR&
   operator=( const CSR< Device_, Index_, IndexAllocator_ >& segments );

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

}  // namespace TNL::Algorithms::Segments

#include "CSR.hpp"
