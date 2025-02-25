// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "BiEllpackBase.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e BiEllpackView is provides a non-owning encapsulation of meta-data stored in
 * the \ref TNL::Algorithms::Segments::BiEllpack segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam Organization is the organization of the elements in the segments—either row-major or column-major order.
 * \tparam WarpSize is the warp size used for the segments.
 */
template< typename Device,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int WarpSize = Backend::getWarpSize() >
class BiEllpackView : public BiEllpackBase< Device, Index, Organization, WarpSize >
{
   using Base = BiEllpackBase< Device, Index, Organization, WarpSize >;

public:
   //! \brief Type of segments view.
   using ViewType = BiEllpackView;

   //! \brief Type of constant segments view.
   using ConstViewType = BiEllpackView< Device, std::add_const_t< Index >, Organization, WarpSize >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = BiEllpackView< Device_, Index_, Organization, WarpSize >;

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   BiEllpackView() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   BiEllpackView( const BiEllpackView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   BiEllpackView( BiEllpackView&& ) noexcept = default;

   //! \brief Constructor that initializes segments based on all necessary data.
   __cuda_callable__
   BiEllpackView( Index size,
                  Index storageSize,
                  typename Base::OffsetsView segmentsPermutation,
                  typename Base::OffsetsView groupPointers );

   //! \brief Copy-assignment operator.
   BiEllpackView&
   operator=( const BiEllpackView& ) = delete;

   //! \brief Move-assignment operator.
   BiEllpackView&
   operator=( BiEllpackView&& ) = delete;

   //! \brief Method for rebinding (reinitialization) to another view.
   __cuda_callable__
   void
   bind( BiEllpackView view );

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
 * \brief Data structure for row-major BiEllpack segments view.
 *
 * See \ref TNL::Algorithms::Segments::BiEllpackView for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 */
template< typename Device, typename Index >
struct RowMajorBiEllpackView : public BiEllpackView< Device, Index, RowMajorOrder >
{
   using BaseType = BiEllpackView< Device, Index, RowMajorOrder >;

   //! \brief Constructor with no parameters to create empty segments.
   RowMajorBiEllpackView() = default;

   //! \brief Copy constructor (makes deep copy).
   RowMajorBiEllpackView( const RowMajorBiEllpackView& );

   //! \brief Move constructor.
   RowMajorBiEllpackView( RowMajorBiEllpackView&& ) noexcept = default;
};

/**
 * \brief Data structure for column-major BiEllpack segments view.
 *
 * See \ref TNL::Algorithms::Segments::BiEllpackView for more details.
 *
 * \tparam Device The type of device on which the segments will operate.
 * \tparam Index The type used for indexing elements managed by the segments.
 */
template< typename Device, typename Index >
struct ColumnMajorBiEllpackView : public BiEllpackView< Device, Index, ColumnMajorOrder >
{
   using BaseType = BiEllpackView< Device, Index, ColumnMajorOrder >;

   //! \brief Constructor with no parameters to create empty segments.
   ColumnMajorBiEllpackView() = default;

   //! \brief Copy constructor (makes deep copy).
   ColumnMajorBiEllpackView( const ColumnMajorBiEllpackView& );

   //! \brief Move constructor.
   ColumnMajorBiEllpackView( ColumnMajorBiEllpackView&& ) noexcept = default;
};

}  // namespace TNL::Algorithms::Segments

#include "BiEllpackView.hpp"
