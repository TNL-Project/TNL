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

   [[nodiscard]] static TNL::String
   getKernelType();

   template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
   static void
   reduceSegments( const SegmentsView& segments,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero );

   template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
   static void
   reduceAllSegments( const SegmentsView& segments,
                      Fetch& fetch,
                      const Reduction& reduction,
                      ResultKeeper& keeper,
                      const Real& zero );
};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "SlicedEllpackKernel.hpp"
