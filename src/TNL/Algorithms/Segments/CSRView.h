// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/SortedSegmentsView.h>
#include "CSRBase.h"
#include "SortedSegmentsView.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e CSRView is provides a non-owning encapsulation of meta-data stored in
 * the \ref TNL::Algorithms::Segments::CSR segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 */
template< typename Device, typename Index >
class CSRView : public CSRBase< Device, Index >
{
   using Base = CSRBase< Device, Index >;

public:
   //! \brief Type of segments view.
   using ViewType = CSRView;

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

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   CSRView() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   CSRView( const CSRView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   CSRView( CSRView&& ) noexcept = default;

   //! \brief Binds a new CSR view to an offsets vector.
   __cuda_callable__
   CSRView( typename Base::OffsetsView offsets );

   //! \brief Copy-assignment operator.
   CSRView&
   operator=( const CSRView& ) = delete;

   //! \brief Move-assignment operator.
   CSRView&
   operator=( CSRView&& ) = delete;

   //! \brief Method for rebinding (reinitialization) to another view.
   __cuda_callable__
   void
   bind( CSRView view );

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
 * \brief Alias for sorted segments based on CSR segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 */
template< typename Device, typename Index >
using SortedCSRView = SortedSegmentsView< CSRView< Device, Index > >;

}  // namespace TNL::Algorithms::Segments

#include "CSRView.hpp"
