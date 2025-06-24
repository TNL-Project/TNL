// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SortedSegmentsBase.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e SortedSegmentsView is provides a non-owning encapsulation of meta-data stored in
 * the \ref TNL::Algorithms::Segments::SortedSegments segments.
 *
 * \tparam EmbeddedSegments is a type of segments used to manage the data.
 */
template< typename EmbeddedSegmentsView_, typename Index = typename EmbeddedSegmentsView_::IndexType >
class SortedSegmentsView : public SortedSegmentsBase< EmbeddedSegmentsView_, Index >
{
   using Base = SortedSegmentsBase< EmbeddedSegmentsView_, Index >;

public:
   using typename Base::EmbeddedSegmentsView;

   using typename Base::EmbeddedSegmentsConstView;

   using typename Base::PermutationView;

   using typename Base::ConstPermutationView;

   //! \brief Type of segments view.
   using ViewType = SortedSegmentsView;

   //! \brief Type of constant segments view.
   using ConstViewType = SortedSegmentsView< EmbeddedSegmentsConstView, std::add_const_t< Index > >;

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   SortedSegmentsView() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   SortedSegmentsView( const SortedSegmentsView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   SortedSegmentsView( SortedSegmentsView&& ) noexcept = default;

   //! \brief Constructor with embedded segments view and segments permutation.
   __cuda_callable__
   SortedSegmentsView( typename Base::EmbeddedSegmentsView embeddedSegmentsView,
                       typename Base::PermutationView segmentsPermutation,
                       typename Base::PermutationView inverseSegmentsPermutation );

   //! \brief Copy-assignment operator.
   SortedSegmentsView&
   operator=( const SortedSegmentsView& ) = delete;

   //! \brief Move-assignment operator.
   SortedSegmentsView&
   operator=( SortedSegmentsView&& ) = delete;

   //! \brief Method for rebinding (reinitialization) to another view.
   __cuda_callable__
   void
   bind( const SortedSegmentsView& view );

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

}  // namespace TNL::Algorithms::Segments

#include "SortedSegmentsView.hpp"
