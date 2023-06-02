// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/String.h>

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

   [[nodiscard]] static TNL::String
   getKernelType();

   template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
   static void
   reduceSegments( const SegmentsView& segments,
                   Index begin,
                   Index end,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Value& identity );

   template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
   static void
   reduceAllSegments( const SegmentsView& segments,
                      Fetch& fetch,
                      const Reduction& reduction,
                      ResultKeeper& keeper,
                      const Value& identity );
};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "CSRVectorKernel.hpp"
