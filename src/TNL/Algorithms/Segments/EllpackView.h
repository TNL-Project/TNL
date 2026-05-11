// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "EllpackBase.h"
#include "SortedSegmentsView.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e EllpackView is provides a non-owning encapsulation of meta-data stored in
 * the \ref TNL::Algorithms::Segments::Ellpack segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam Organization is the organization of the elements in the segments—either row-major or column-major order.
 * \tparam Alignment is the alignment of the number of segments (to optimize data alignment, particularly on GPUs).
 */
template< typename Device,
          typename Index,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
class EllpackView : public EllpackBase< Device, Index, Organization, Alignment >
{
   using Base = EllpackBase< Device, Index, Organization, Alignment >;

public:
   //! \brief Type of segments view.
   using ViewType = EllpackView;

   //! \brief Type of constant segments view.
   using ConstViewType = ViewType;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = EllpackView< Device_, Index_, Organization, Alignment >;

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   EllpackView() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   EllpackView( const EllpackView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   EllpackView( EllpackView&& ) noexcept = default;

   //! \brief Constructor that initializes segments based on their sizes.
   __cuda_callable__
   EllpackView( Index segmentsCount, Index segmentSize, Index alignedSize );

   //! \brief Constructor that initializes segments based on their sizes.
   __cuda_callable__
   EllpackView( Index segmentsCount, Index segmentSize );

   //! \brief Copy-assignment operator.
   EllpackView&
   operator=( const EllpackView& ) = delete;

   //! \brief Move-assignment operator.
   EllpackView&
   operator=( EllpackView&& ) = delete;

   //! \brief Method for rebinding (reinitialization) to another view.
   __cuda_callable__
   void
   bind( EllpackView view );

   //! \brief Returns a view for this instance of segments which can by used
   //! for example in lambda functions running in GPU kernels.
   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   //! \brief Returns a constant view for this instance of segments which
   //! can by used for example in lambda functions running in GPU kernels.
   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;

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
 * \brief Alias for row-major Ellpack segments view.
 *
 * See \ref TNL::Algorithms::Segments::EllpackView for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Alignment The alignment of the number of segments (to optimize data
 * alignment, particularly on GPUs).
 */
template< typename Device, typename Index, int Alignment = 32 >
using RowMajorEllpackView = EllpackView< Device, Index, RowMajorOrder, Alignment >;

/**
 * \brief Alias for column-major Ellpack segments view.
 *
 * See \ref TNL::Algorithms::Segments::EllpackView for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam Alignment The alignment of the number of segments (to optimize data
 * alignment, particularly on GPUs).
 */
template< typename Device, typename Index, int Alignment = 32 >
using ColumnMajorEllpackView = EllpackView< Device, Index, ColumnMajorOrder, Alignment >;

/**
 * \brief Alias for sorted segments based on EllpackView segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device,
          typename Index,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
using SortedEllpackView = SortedSegmentsView< EllpackView< Device, Index, Organization, Alignment > >;

/**
 * \brief Alias for sorted segments based on row-major EllpackView segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device, typename Index, int Alignment = 32 >
using SortedRowMajorEllpackView = SortedSegmentsView< RowMajorEllpackView< Device, Index, Alignment > >;

/**
 * \brief Alias for sorted segments based on column-major Ellpack segments.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device, typename Index, int Alignment = 32 >
using SortedColumnMajorEllpackView = SortedSegmentsView< ColumnMajorEllpackView< Device, Index, Alignment > >;

}  // namespace TNL::Algorithms::Segments

#include "EllpackView.hpp"
