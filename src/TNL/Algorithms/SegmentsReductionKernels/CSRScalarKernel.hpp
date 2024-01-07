// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "CSRScalarKernel.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

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
std::string
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

   auto l = [ offsets, fetch, reduction, keeper, identity ] __cuda_callable__( const Index segmentIdx ) mutable
   {
      const Index begin = offsets[ segmentIdx ];
      const Index end = offsets[ segmentIdx + 1 ];
      using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
      ReturnType aux = identity;
      bool compute = true;
      if constexpr( detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() ) {
         Index localIdx = 0;
         for( Index globalIdx = begin; globalIdx < end && compute; globalIdx++ )
            aux = reduction( aux, fetch( segmentIdx, localIdx++, globalIdx, compute ) );
      }
      else {
         for( Index globalIdx = begin; globalIdx < end && compute; globalIdx++ )
            aux = reduction( aux, fetch( globalIdx, compute ) );
      }
      keeper( segmentIdx, aux );
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
