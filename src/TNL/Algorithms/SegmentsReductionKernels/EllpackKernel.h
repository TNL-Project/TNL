// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include "isSegmentReductionKernel.h"

#include "../Segments/detail/FetchLambdaAdapter.h"
#include "isSegmentsReductionKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Device >
struct EllpackKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = EllpackKernel< Index, Device >;
   using ConstViewType = EllpackKernel< Index, Device >;

   template< typename Segments >
   void
   init( const Segments& segments );

   void
   reset();

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
             typename Value = typename Segments::detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   [[deprecated( "Use TNL::Algorithms::Segments::reduceSegments instead" )]] static void
   reduceSegments( const SegmentsView& segments,
                   Index begin,
                   Index end,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Value& identity = Reduction::template getIdentity< Value >() );

   template< typename SegmentsView,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename Segments::detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   [[deprecated( "Use TNL::Algorithms::Segments::reduceSegments instead" )]] static void
   reduceAllSegments( const SegmentsView& segments,
                      Fetch& fetch,
                      const Reduction& reduction,
                      ResultKeeper& keeper,
                      const Value& identity = Reduction::template getIdentity< Value >() );
};

template< typename Index, typename Device >
struct isSegmentsReductionKernel< EllpackKernel< Index, Device > > : std::true_type
{};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "EllpackKernel.hpp"
