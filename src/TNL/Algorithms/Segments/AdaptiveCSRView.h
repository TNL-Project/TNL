// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "CSRView.h"
#include "detail/CSRAdaptiveKernelBlockDescriptor.h"
#include "detail/CSRAdaptiveKernelParameters.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Data structure for adaptive CSR segments format.
 *
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam IndexAllocator is allocator for supporting index containers.
 */
template< typename Device, typename Index >
class AdaptiveCSRView : public CSRView< Device, Index >
{
   using Base = CSRView< Device, Index >;

public:
   //! \brief Type of segments view.
   using ViewType = AdaptiveCSRView< Device, Index >;

   //! \brief Type of constant segments view.
   using ConstViewType = AdaptiveCSRView< Device, std::add_const_t< Index > >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = AdaptiveCSRView< Device_, Index_ >;

   using OffsetsView = typename Base::OffsetsView;

   using BlocksType = TNL::Containers::Vector< detail::CSRAdaptiveKernelBlockDescriptor< Index >, Device, Index >;
   using BlocksView = typename BlocksType::ViewType;

   [[nodiscard]] static constexpr int
   MaxValueSizeLog()
   {
      return detail::CSRAdaptiveKernelParameters<>::MaxValueSizeLog;
   }

   [[nodiscard]] static int
   getSizeValueLog( const int& i )
   {
      return detail::CSRAdaptiveKernelParameters<>::getSizeValueLog( i );
   }

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   AdaptiveCSRView() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   AdaptiveCSRView( const AdaptiveCSRView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   AdaptiveCSRView( AdaptiveCSRView&& ) noexcept = default;

   __cuda_callable__
   AdaptiveCSRView( const CSRView< Device, Index >& csrView, BlocksView* blocksView );

   __cuda_callable__
   AdaptiveCSRView( const CSRView< Device, Index >& csrView, BlocksType* blocksView );

   AdaptiveCSRView&
   operator=( const AdaptiveCSRView< Index, Device >& kernelView ) = delete;

   AdaptiveCSRView&
   operator=( const AdaptiveCSRView< Index, Device >&& kernelView ) = delete;

   //! \brief Method for rebinding (reinitialization) to another view.
   __cuda_callable__
   void
   bind( AdaptiveCSRView view );

   __cuda_callable__
   void
   bind( OffsetsView offsets, BlocksView* blocks );

   __cuda_callable__
   void
   bind( OffsetsView offsets, BlocksType* blocks );

   void
   setBlocks( BlocksType& blocks, int idx );

   /**
    * \brief Returns a view for this instance of CSR segments which can by used
    * for example in lambda functions running in GPU kernels.
    */
   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   /**
    * \brief Returns a constant view for this instance of CSR segments which
    * can by used for example in lambda functions running in GPU kernels.
    */
   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;

   [[nodiscard]] __cuda_callable__
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

   void
   printBlocks( int idx ) const;

protected:
   BlocksView blocksArray[ MaxValueSizeLog() ];
};

}  // namespace TNL::Algorithms::Segments

#include "AdaptiveCSRView.hpp"