// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Segments/detail/LambdaAdapter.h>

namespace TNL::Algorithms::Segments {

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

   template< typename Offsets >
   void
   init( const Offsets& offsets );

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

   [[nodiscard]] TNL::String
   getSetup() const;

   template< typename OffsetsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
   void
   reduceSegments( const OffsetsView& offsets,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero ) const;

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

}  // namespace TNL::Algorithms::Segments

#include <TNL/Algorithms/Segments/Kernels/CSRLightKernel.hpp>
