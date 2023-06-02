// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Algorithms/parallelFor.h>

#include "CSRScalarKernel.h"
#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduction,
          typename Keep,
          bool DispatchScalarCSR = detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() >
struct CSRScalarKernelreduceSegmentsDispatcher;

template< typename Index, typename Device, typename Fetch, typename Reduction, typename ResultKeeper >
struct CSRScalarKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduction, ResultKeeper, true >
{
   template< typename Offsets, typename Value >
   static void
   reduce( const Offsets& offsets,
           Index begin,
           Index end,
           Fetch& fetch,
           const Reduction& reduction,
           ResultKeeper& keep,
           const Value& identity )
   {
      auto l = [ = ] __cuda_callable__( const Index segmentIdx ) mutable
      {
         const Index begin = offsets[ segmentIdx ];
         const Index end = offsets[ segmentIdx + 1 ];
         using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
         ReturnType aux = identity;
         Index localIdx = 0;
         bool compute = true;
         for( Index globalIdx = begin; globalIdx < end && compute; globalIdx++ )
            aux = reduction( aux, fetch( segmentIdx, localIdx++, globalIdx, compute ) );
         keep( segmentIdx, aux );
      };

      if constexpr( std::is_same< Device, TNL::Devices::Sequential >::value ) {
         for( Index segmentIdx = begin; segmentIdx < end; segmentIdx++ )
            l( segmentIdx );
      }
      else if constexpr( std::is_same< Device, TNL::Devices::Host >::value ) {
#ifdef HAVE_OPENMP
         #pragma omp parallel for firstprivate( l ) schedule( dynamic, 100 ), if( Devices::Host::isOMPEnabled() )
#endif
         for( Index segmentIdx = begin; segmentIdx < end; segmentIdx++ )
            l( segmentIdx );
      }
      else
         Algorithms::parallelFor< Device >( begin, end, l );
   }
};

template< typename Index, typename Device, typename Fetch, typename Reduction, typename Keep >
struct CSRScalarKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduction, Keep, false >
{
   template< typename OffsetsView, typename Value >
   static void
   reduce( const OffsetsView& offsets,
           Index begin,
           Index end,
           Fetch& fetch,
           const Reduction& reduction,
           Keep& keep,
           const Value& identity )
   {
      auto l = [ = ] __cuda_callable__( const Index segmentIdx ) mutable
      {
         const Index begin = offsets[ segmentIdx ];
         const Index end = offsets[ segmentIdx + 1 ];
         using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
         ReturnType aux = identity;
         bool compute = true;
         for( Index globalIdx = begin; globalIdx < end && compute; globalIdx++ )
            aux = reduction( aux, fetch( globalIdx, compute ) );
         keep( segmentIdx, aux );
      };

      if constexpr( std::is_same< Device, TNL::Devices::Sequential >::value ) {
         for( Index segmentIdx = begin; segmentIdx < end; segmentIdx++ )
            l( segmentIdx );
      }
      else if constexpr( std::is_same< Device, TNL::Devices::Host >::value ) {
#ifdef HAVE_OPENMP
         #pragma omp parallel for firstprivate( l ) schedule( dynamic, 100 ), if( Devices::Host::isOMPEnabled() )
#endif
         for( Index segmentIdx = begin; segmentIdx < end; segmentIdx++ )
            l( segmentIdx );
      }
      else
         Algorithms::parallelFor< Device >( begin, end, l );
   }
};

template< typename Index, typename Device >
template< typename Segments >
void
CSRScalarKernel< Index, Device >::init( const Segments& segments )
{}

template< typename Index, typename Device >
void
CSRScalarKernel< Index, Device >::reset()
{}

template< typename Index, typename Device >
__cuda_callable__
auto
CSRScalarKernel< Index, Device >::getView() -> ViewType
{
   return *this;
}

template< typename Index, typename Device >
__cuda_callable__
auto
CSRScalarKernel< Index, Device >::getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index, typename Device >
TNL::String
CSRScalarKernel< Index, Device >::getKernelType()
{
   return "Scalar";
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRScalarKernel< Index, Device >::reduceSegments( const SegmentsView& segments,
                                                  Index begin,
                                                  Index end,
                                                  Fetch& fetch,
                                                  const Reduction& reduction,
                                                  ResultKeeper& keeper,
                                                  const Value& identity )
{
   using OffsetsView = typename SegmentsView::ConstOffsetsView;
   OffsetsView offsets = segments.getOffsets();

   CSRScalarKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduction, ResultKeeper >::reduce(
      offsets, begin, end, fetch, reduction, keeper, identity );
}

template< typename Index, typename Device >
template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
void
CSRScalarKernel< Index, Device >::reduceAllSegments( const SegmentsView& segments,
                                                     Fetch& fetch,
                                                     const Reduction& reduction,
                                                     ResultKeeper& keeper,
                                                     const Value& identity )
{
   reduceSegments( segments, 0, segments.getSegmentsCount(), fetch, reduction, keeper, identity );
}

}  // namespace TNL::Algorithms::SegmentsReductionKernels
