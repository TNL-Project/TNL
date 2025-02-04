// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
#include "isSegmentReductionKernel.h"

#include "detail/CSRAdaptiveKernelBlockDescriptor.h"
#include "detail/CSRAdaptiveKernelParameters.h"
#include "detail/FetchLambdaAdapter.h"
#include "isSegmentsReductionKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Device >
struct CSRAdaptiveKernelView
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRAdaptiveKernelView< Index, Device >;
   using ConstViewType = CSRAdaptiveKernelView< Index, Device >;
   using BlocksType = TNL::Containers::Vector< detail::CSRAdaptiveKernelBlockDescriptor< Index >, Device, Index >;
   using BlocksView = typename BlocksType::ViewType;

   static constexpr int MaxValueSizeLog = detail::CSRAdaptiveKernelParameters<>::MaxValueSizeLog;

   [[nodiscard]] static int
   getSizeValueLog( const int& i )
   {
      return detail::CSRAdaptiveKernelParameters<>::getSizeValueLog( i );
   }

   CSRAdaptiveKernelView() = default;

   void
   setBlocks( BlocksType& blocks, int idx );

   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;

   [[nodiscard]] static std::string
   getKernelType();

   template< typename SegmentsView,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   void
   reduceSegments( const SegmentsView& segments,
                   Index begin,
                   Index end,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Value& identity = Reduction::template getIdentity< Value >() ) const;

   template< typename SegmentsView,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   void
   reduceAllSegments( const SegmentsView& segments,
                      Fetch& fetch,
                      const Reduction& reduction,
                      ResultKeeper& keeper,
                      const Value& identity = Reduction::template getIdentity< Value >() ) const;

   CSRAdaptiveKernelView&
   operator=( const CSRAdaptiveKernelView< Index, Device >& kernelView ) = delete;

   void
   printBlocks( int idx ) const;

protected:
   BlocksView blocksArray[ MaxValueSizeLog ];
};

template< typename Index, typename Device >
struct isSegmentsReductionKernel< CSRAdaptiveKernelView< Index, Device > > : std::true_type
{};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "CSRAdaptiveKernelView.hpp"
