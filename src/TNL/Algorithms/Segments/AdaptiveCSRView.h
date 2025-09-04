// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "CSRView.h"
#include "detail/CSRAdaptiveKernelBlockDescriptor.h"
#include "detail/CSRAdaptiveKernelParameters.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief \e AdaptiveCSRView is provides a non-owning encapsulation of meta-data stored in
 * the AdaptiveCSR segments.
 *
 * See \ref TNL::Algorithms::Segments::AdaptiveCSR for more details about AdaptiveCSR segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 */
template< typename Device, typename Index >
class AdaptiveCSRView : public CSRView< Device, Index >
{
   using Base = CSRView< Device, Index >;

public:
   using typename Base::DeviceType;
   using typename Base::IndexType;

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

   using BlocksType = TNL::Containers::Vector< detail::CSRAdaptiveKernelBlockDescriptor< IndexType >, Device, IndexType >;
   using BlocksView = typename BlocksType::ViewType;

   //! \brief Default constructor with no parameters to create empty segments view.
   __cuda_callable__
   AdaptiveCSRView() = default;

   //! \brief Copy constructor.
   __cuda_callable__
   AdaptiveCSRView( const AdaptiveCSRView& ) = default;

   //! \brief Move constructor.
   __cuda_callable__
   AdaptiveCSRView( AdaptiveCSRView&& ) noexcept = default;

   //! \brief Binds a new CSR view together with blocks of AdaptiveCSR.
   __cuda_callable__
   AdaptiveCSRView( const CSRView< Device, Index >& csrView, BlocksView* blocksView );

   //! \brief Binds a new CSR view together with blocks of AdaptiveCSR.
   __cuda_callable__
   AdaptiveCSRView( const CSRView< Device, Index >& csrView, BlocksType* blocksView );

   //! \brief Copy-assignment operator.
   AdaptiveCSRView&
   operator=( const AdaptiveCSRView< Index, Device >& kernelView ) = delete;

   //! \brief Move-assignment operator.
   AdaptiveCSRView&
   operator=( const AdaptiveCSRView< Index, Device >&& kernelView ) = delete;

   //! \brief Method for rebinding (reinitialization) to another view.
   __cuda_callable__
   void
   bind( AdaptiveCSRView view );

   //! \brief Method for rebinding (reinitialization) using another CSR offsets and AdaptiveCSR blocks.
   __cuda_callable__
   void
   bind( OffsetsView offsets, BlocksView* blocks );

   //! \brief Method for rebinding (reinitialization) using another CSR offsets and AdaptiveCSR blocks.
   __cuda_callable__
   void
   bind( OffsetsView offsets, BlocksType* blocks );

   //! \brief Method for setting AdaptiveCSR blocks.
   void
   setBlocks( BlocksType& blocks, int idx );

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
   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   //! \brief Returns a constant view for this instance of segments which
   //! can by used for example in lambda functions running in GPU kernels.
   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;

   //! \brief Returns a view with blocks of AdaptiveCSR.
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

   //! \brief Print the blocks of AdaptiveCSR.
   void
   printBlocks( int idx = 1 ) const;

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

protected:
   BlocksView blocksArray[ MaxValueSizeLog() ];
};

/**
 * \brief Alias for sorted segments based on CSR segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 */
template< typename Device, typename Index >
using SortedAdaptiveCSRView = SortedSegmentsView< AdaptiveCSRView< Device, Index > >;

}  // namespace TNL::Algorithms::Segments

#include "AdaptiveCSRView.hpp"
