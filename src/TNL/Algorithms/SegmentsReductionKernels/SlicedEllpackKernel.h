// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>

#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Device >
struct SlicedEllpackKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = SlicedEllpackKernel< Index, Device >;
   using ConstViewType = SlicedEllpackKernel< Index, Device >;

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
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
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
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceAllSegments( const SegmentsView& segments,
                      Fetch& fetch,
                      const Reduction& reduction,
                      ResultKeeper& keeper,
                      const Value& identity = Reduction::template getIdentity< Value >() );
};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "SlicedEllpackKernel.hpp"
