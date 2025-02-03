// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include "isSegmentReductionKernel.h"

#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Device >
struct CSRVectorKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRVectorKernel< Index, Device >;
   using ConstViewType = CSRVectorKernel< Index, Device >;

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
   [[deprecated( "Use TNL::Algorithms::Segments::reduceSegments instead" )]]
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
   [[deprecated( "Use TNL::Algorithms::Segments::reduceAllSegments instead" )]]
   static void
   reduceAllSegments( const SegmentsView& segments,
                      Fetch& fetch,
                      const Reduction& reduction,
                      ResultKeeper& keeper,
                      const Value& identity = Reduction::template getIdentity< Value >() );
};

template< typename Index, typename Device >
struct isSegmentReductionKernel< CSRVectorKernel< Index, Device > >
{
   static constexpr bool value = true;
};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "CSRVectorKernel.hpp"
