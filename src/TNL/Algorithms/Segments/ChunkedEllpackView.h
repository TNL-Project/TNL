// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "ChunkedEllpackBase.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e ChunkedView is provides a non-owning encapsulation of meta-data stored in
 * the \ref TNL::Algorithms::Segments::ChunkedEllpack segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam Organization is the organization of the elements in the segments—either row-major or column-major order.
 */
template< typename Device,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class ChunkedEllpackView : public ChunkedEllpackBase< Device, Index, Organization >
{
   using Base = ChunkedEllpackBase< Device, Index, Organization >;

public:
   //! \brief Type of segments view.
   using ViewType = ChunkedEllpackView;

   //! \brief Type of constant segments view.
   using ConstViewType = ChunkedEllpackView< Device, std::add_const_t< Index >, Organization >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = ChunkedEllpackView< Device_, Index_, Organization >;

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   ChunkedEllpackView() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   ChunkedEllpackView( const ChunkedEllpackView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   ChunkedEllpackView( ChunkedEllpackView&& ) noexcept = default;

   //! \brief Constructor that initializes segments based on all necessary data.
   __cuda_callable__
   ChunkedEllpackView( Index size,
                       Index storageSize,
                       Index numberOfSlices,
                       Index chunksInSlice,
                       Index desiredChunkSize,
                       typename Base::OffsetsView segmentToChunkMapping,
                       typename Base::OffsetsView segmentToSliceMapping,
                       typename Base::OffsetsView chunksToSegmentsMapping,
                       typename Base::OffsetsView segmentPointers,
                       typename Base::SliceInfoContainerView slices );

   //! \brief Copy-assignment operator.
   ChunkedEllpackView&
   operator=( const ChunkedEllpackView& ) = delete;

   //! \brief Move-assignment operator.
   ChunkedEllpackView&
   operator=( ChunkedEllpackView&& ) = delete;

   //! \brief Method for rebinding (reinitialization) to another view.
   __cuda_callable__
   void
   bind( ChunkedEllpackView view );

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
 * \brief Data structure for row-major Chunked Ellpack segments view.
 *
 * See \ref TNL::Algorithms::Segments::ChunkedEllpackView for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 */
template< typename Device, typename Index >
struct RowMajorChunkedEllpackView : public ChunkedEllpackView< Device, Index, RowMajorOrder >
{
   using BaseType = ChunkedEllpackView< Device, Index, RowMajorOrder >;

   //! \brief Constructor with no parameters to create empty segments.
   RowMajorChunkedEllpackView() = default;

   //! \brief Copy constructor (makes deep copy).
   RowMajorChunkedEllpackView( const RowMajorChunkedEllpackView& );

   //! \brief Move constructor.
   RowMajorChunkedEllpackView( RowMajorChunkedEllpackView&& ) noexcept = default;
};

/**
 * \brief Data structure for column-major Chunked Ellpack segments view.
 *
 * See \ref TNL::Algorithms::Segments::ChunkedEllpackView for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 */
template< typename Device, typename Index >
struct ColumnMajorChunkedEllpackView : public ChunkedEllpackView< Device, Index, ColumnMajorOrder >
{
   using BaseType = ChunkedEllpackView< Device, Index, ColumnMajorOrder >;

   //! \brief Constructor with no parameters to create empty segments.
   ColumnMajorChunkedEllpackView() = default;

   //! \brief Copy constructor (makes deep copy).
   ColumnMajorChunkedEllpackView( const ColumnMajorChunkedEllpackView& );

   //! \brief Move constructor.
   ColumnMajorChunkedEllpackView( ColumnMajorChunkedEllpackView&& ) noexcept = default;
};

}  // namespace TNL::Algorithms::Segments

#include "ChunkedEllpackView.hpp"
