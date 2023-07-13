// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Cuda/LaunchHelpers.h>

#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Index, typename Device >
struct CSRScalarKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRScalarKernel< Index, Device >;
   using ConstViewType = CSRScalarKernel< Index, Device >;

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

   /**
    * \brief Compute reduction in each segment.
    *
    * \tparam Fetch is type of lambda function for data fetching.
    * \tparam Reduction is a reduction operation.
    * \tparam ResultKeeper is lambda function for storing results from particular segments.
    *
    * \param segments is the segments data structure to be reduced.
    * \param begin defines begining of an interval [ \e begin, \e end ) of segments in
    *    which we want to perform the reduction.
    * \param end defines and of an interval [ \e begin, \e end ) of segments in
    *    which we want to perform the reduction.
    * \param fetch is a lambda function for fetching of data. It is suppos have one of the
    *  following forms:
    * 1. Full form
    *  ```
    *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, bool& compute ) { ...
    * }
    *  ```
    * 2. Brief form
    * ```
    * auto fetch = [=] __cuda_callable__ ( IndexType globalIdx, bool& compute ) { ... }
    * ```
    * where for both variants \e segmentIdx is segment index, \e localIdx is a
    * rank of element in the segment, \e globalIdx is index of the element in
    * related container and \e compute is a boolean variable which serves for
    * stopping the reduction if it is set to \e false. It is however, only a
    * hint and the real behaviour depends on type of kernel used for the
    * reduction.  Some kernels are optimized so that they can be significantly
    * faster with the brief variant of the \e fetch lambda function.
    *
    * \param reduction is a lambda function representing the reduction opeartion. It is
    * supposed to be defined as:
    *
    * ```
    * auto reduction = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
    * ```
    *
    * where \e a and \e b are values to be reduced and the lambda function returns result of the reduction.
    * \param keeper is a lambda function for saving results from particular segments. It is supposed to be defined as:
    *
    * ```
    * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
    * ```
    *
    * where \e segmentIdx is an index of the segment and \e value is the result of the reduction in given segment to be stored.
    *
    * \param identity is the initial value for the reduction operation.
    *                 If \e Reduction does not have a static member function
    *                 template \e getIdentity, it must be supplied explicitly
    *                 by the user.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_reduceSegments.cpp
    * \par Output
    * \include SegmentsExample_CSR_reduceSegments.out
    */
   template< typename SegmentsView,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegments( const SegmentsView& segments,
                   Index begin,
                   Index end,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Value& identity = Reduction::template getIdentity< Value >() );

   /**
    * \brief Call \ref reduceSegments for all segments.
    *
    * See \ref reduceSegments for more details.
    */
   template< typename SegmentsView,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceAllSegments( const SegmentsView& segments,
                      Fetch& fetch,
                      const Reduction& reduction,
                      ResultKeeper& keeper,
                      const Value& identity = Reduction::template getIdentity< Value >() );
};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "CSRScalarKernel.hpp"
