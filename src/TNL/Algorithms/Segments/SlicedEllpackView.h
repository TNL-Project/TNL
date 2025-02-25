// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SlicedEllpackBase.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e SlicedEllpackView is provides a non-owning encapsulation of meta-data stored in
 * the \ref TNL::Algorithms::Segments::SlicedEllpack segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam Organization is the organization of the elements in the segments—either row-major or column-major order.
 * \tparam SliceSize is the size of each slice.
 */
template< typename Device,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int SliceSize = 32 >
class SlicedEllpackView : public SlicedEllpackBase< Device, Index, Organization, SliceSize >
{
   using Base = SlicedEllpackBase< Device, Index, Organization, SliceSize >;

public:
   //! \brief Type of segments view.
   using ViewType = SlicedEllpackView;

   //! \brief Type of constant segments view.
   using ConstViewType = SlicedEllpackView< Device, std::add_const_t< Index >, Organization, SliceSize >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = SlicedEllpackView< Device_, Index_, Organization, SliceSize >;

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   SlicedEllpackView() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   SlicedEllpackView( const SlicedEllpackView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   SlicedEllpackView( SlicedEllpackView&& ) noexcept = default;

   //! \brief Constructor that initializes segments based all necessary data.
   __cuda_callable__
   SlicedEllpackView( Index size,
                      Index alignedSize,
                      Index segmentsCount,
                      typename Base::OffsetsView sliceOffsets,
                      typename Base::OffsetsView sliceSegmentSizes );

   //! \brief Copy-assignment operator.
   SlicedEllpackView&
   operator=( const SlicedEllpackView& ) = delete;

   //! \brief Move-assignment operator.
   SlicedEllpackView&
   operator=( SlicedEllpackView&& ) = delete;

   //! \brief Method for rebinding (reinitialization) to another view.
   __cuda_callable__
   void
   bind( SlicedEllpackView view );

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
 * \brief Data structure for row-major SlicedEllpack view.
 *
 * See \ref TNL::Algorithms::Segments::SlicedEllpackView for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 * \tparam SliceSize The size of each slice.
 */
template< typename Device, typename Index, int SliceSize = 32 >
struct RowMajorSlicedEllpackView : public SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >
{
   using BaseType = SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >;

   //! \brief Constructor with no parameters to create empty segments.
   __cuda_callable__
   RowMajorSlicedEllpackView() = default;

   //! \brief Copy constructor (makes deep copy).
   __cuda_callable__
   RowMajorSlicedEllpackView( const RowMajorSlicedEllpackView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   RowMajorSlicedEllpackView( RowMajorSlicedEllpackView&& ) noexcept = default;

   //! \brief Constructor that initializes segments based all necessary data.
   __cuda_callable__
   RowMajorSlicedEllpackView( Index size,
                              Index alignedSize,
                              Index segmentsCount,
                              typename BaseType::OffsetsView sliceOffsets,
                              typename BaseType::OffsetsView sliceSegmentSizes )
   : BaseType( size, alignedSize, segmentsCount, sliceOffsets, sliceSegmentSizes )
   {}
};

/**
 * \brief Data structure for column-major SlicedEllpack view.
 *
 * See \ref TNL::Algorithms::Segments::SlicedEllpackView for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 * \tparam IndexAllocator The allocator used for managing index containers.
 */
template< typename Device, typename Index, int SliceSize = 32 >
struct ColumnMajorSlicedEllpackView : public SlicedEllpackView< Device, Index, ColumnMajorOrder, SliceSize >
{
   using BaseType = SlicedEllpackView< Device, Index, ColumnMajorOrder, SliceSize >;

   //! \brief Constructor with no parameters to create empty segments.
   __cuda_callable__
   ColumnMajorSlicedEllpackView() = default;

   //! \brief Copy constructor (makes deep copy).
   __cuda_callable__
   ColumnMajorSlicedEllpackView( const ColumnMajorSlicedEllpackView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   ColumnMajorSlicedEllpackView( ColumnMajorSlicedEllpackView&& ) noexcept = default;

   //! \brief Constructor that initializes segments based all necessary data.
   __cuda_callable__
   ColumnMajorSlicedEllpackView( Index size,
                                 Index alignedSize,
                                 Index segmentsCount,
                                 typename BaseType::OffsetsView sliceOffsets,
                                 typename BaseType::OffsetsView sliceSegmentSizes )
   : BaseType( size, alignedSize, segmentsCount, sliceOffsets, sliceSegmentSizes )
   {}
};

}  // namespace TNL::Algorithms::Segments

#include "SlicedEllpackView.hpp"
