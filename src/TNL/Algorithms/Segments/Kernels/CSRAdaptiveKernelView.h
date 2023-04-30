// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/Kernels/details/CSRAdaptiveKernelBlockDescriptor.h>
#include <TNL/Algorithms/Segments/Kernels/details/CSRAdaptiveKernelParameters.h>

namespace TNL::Algorithms::Segments {

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

   static int
   getSizeValueLog( const int& i )
   {
      return detail::CSRAdaptiveKernelParameters<>::getSizeValueLog( i );
   }

   CSRAdaptiveKernelView() = default;

   void
   setBlocks( BlocksType& blocks, int idx );

   __cuda_callable__
   ViewType
   getView();

   __cuda_callable__
   ConstViewType
   getConstView() const;

   static TNL::String
   getKernelType();

   template< typename OffsetsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
   void
   reduceSegments( const OffsetsView& offsets,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero,
                   Args... args ) const;

   CSRAdaptiveKernelView&
   operator=( const CSRAdaptiveKernelView< Index, Device >& kernelView );

   void
   printBlocks( int idx ) const;

protected:
   BlocksView blocksArray[ MaxValueSizeLog ];
};

}  // namespace TNL::Algorithms::Segments

#include <TNL/Algorithms/Segments/Kernels/CSRAdaptiveKernelView.hpp>
