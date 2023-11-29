// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>

#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

enum LightCSRSThreadsMapping
{
   LightCSRConstantThreads,
   CSRLightAutomaticThreads,
   CSRLightAutomaticThreadsLightSpMV
};

template< typename Index, typename Device >
struct CSRLightKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRLightKernel< Index, Device >;
   using ConstViewType = CSRLightKernel< Index, Device >;

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

   [[nodiscard]] std::string
   getSetup() const;

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

   void
   setThreadsMapping( LightCSRSThreadsMapping mapping );

   [[nodiscard]] LightCSRSThreadsMapping
   getThreadsMapping() const;

   void
   setThreadsPerSegment( int threadsPerSegment );

   [[nodiscard]] int
   getThreadsPerSegment() const;

protected:
   LightCSRSThreadsMapping mapping = CSRLightAutomaticThreads;

   int threadsPerSegment = 32;
};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "CSRLightKernel.hpp"
