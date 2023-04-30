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
#include <TNL/Algorithms/Segments/Kernels/CSRScalarKernel.h>
#include <TNL/Algorithms/Segments/Kernels/CSRAdaptiveKernelView.h>
#include <TNL/Algorithms/Segments/Kernels/details/CSRAdaptiveKernelBlockDescriptor.h>

namespace TNL::Algorithms::Segments {

#ifdef __CUDACC__

template< int CudaBlockSize,
          int warpSize,
          int WARPS,
          int SHARED_PER_WARP,
          int MAX_ELEM_PER_WARP,
          typename BlocksView,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__
void
reduceSegmentsCSRAdaptiveKernel( BlocksView blocks,
                                 int gridIdx,
                                 Offsets offsets,
                                 Index first,
                                 Index last,
                                 Fetch fetch,
                                 Reduction reduce,
                                 ResultKeeper keep,
                                 Real zero,
                                 Args... args );
#endif

template< typename Index, typename Device >
struct CSRAdaptiveKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRAdaptiveKernelView< Index, Device >;
   using ConstViewType = CSRAdaptiveKernelView< Index, Device >;
   using BlocksType = typename ViewType::BlocksType;
   using BlocksView = typename BlocksType::ViewType;

   [[nodiscard]] static constexpr int
   MaxValueSizeLog()
   {
      return ViewType::MaxValueSizeLog;
   }

   [[nodiscard]] static int
   getSizeValueLog( const int& i )
   {
      return detail::CSRAdaptiveKernelParameters<>::getSizeValueLog( i );
   }

   [[nodiscard]] static TNL::String
   getKernelType();

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

protected:
   template< int SizeOfValue, typename Offsets >
   Index
   findLimit( Index start, const Offsets& offsets, Index size, detail::Type& type, size_t& sum );

   template< int SizeOfValue, typename Offsets >
   void
   initValueSize( const Offsets& offsets );

   /**
    * \brief  blocksArray[ i ] stores blocks for sizeof( Value ) == 2^i.
    */
   BlocksType blocksArray[ MaxValueSizeLog() ];

   ViewType view;
};

}  // namespace TNL::Algorithms::Segments

#include <TNL/Algorithms/Segments/Kernels/CSRAdaptiveKernel.hpp>
