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

   [[nodiscard]] static TNL::String
   getKernelType();

   /**
    * \brief Compute reduction in each segment.
    *
    * \tparam Fetch is type of lambda function for data fetching.
    * \tparam Reduce is a reduction operation.
    * \tparam Keep is lambda function for storing results from particular segments.
    *
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
    * \param reduce is a lambda function representing the reduction opeartion. It is
    * supposed to be defined as:
    *
    * ```
    * auto reduce = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
    * ```
    *
    * where \e a and \e b are values to be reduced and the lambda function returns result of the reduction.
    * \param keep is a lambda function for saving results from particular segments. It is supposed to be defined as:
    *
    * ```
    * auto keep = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
    * ```
    *
    * where \e segmentIdx is an index of the segment and \e value is the result of the reduction in given segment to be stored.
    *
    * \param zero is the initial value for the reduction operation.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_reduceSegments.cpp
    * \par Output
    * \include SegmentsExample_CSR_reduceSegments.out
    */
   template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
   static void
   reduceSegments( const SegmentsView& segments,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero );

   /**
    * \brief Call \ref reduceSegments for all segments.
    *
    * See \ref reduceSegments for more details.
    */
   template< typename SegmentsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
   static void
   reduceAllSegments( const SegmentsView& segments,
                      Fetch& fetch,
                      const Reduction& reduction,
                      ResultKeeper& keeper,
                      const Real& zero );
};

}  // namespace TNL::Algorithms::SegmentsReductionKernels

#include "CSRScalarKernel.hpp"
