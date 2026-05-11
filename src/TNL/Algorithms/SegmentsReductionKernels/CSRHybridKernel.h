// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include "isSegmentReductionKernel.h"

#include "detail/FetchLambdaAdapter.h"
#include "isSegmentsReductionKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Device, int ThreadsInBlock = 128 >
struct CSRHybridKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRHybridKernel< Index, Device, ThreadsInBlock >;
   using ConstViewType = CSRHybridKernel< Index, Device, ThreadsInBlock >;

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
   [[deprecated( "Use TNL::Algorithms::Segments::reduceSegments instead" )]] void
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
   [[deprecated( "Use TNL::Algorithms::Segments::reduceAllSegments instead" )]] void
   reduceAllSegments( const SegmentsView& segments,
                      Fetch& fetch,
                      const Reduction& reduction,
                      ResultKeeper& keeper,
                      const Value& identity = Reduction::template getIdentity< Value >() ) const;

protected:
   int threadsPerSegment = 0;
};

template< typename Index, typename Device >
struct isSegmentsReductionKernel< CSRHybridKernel< Index, Device > > : std::true_type
{};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "CSRHybridKernel.hpp"
