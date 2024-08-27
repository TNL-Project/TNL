// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

#include "CheckLambdas.h"

namespace TNL::Algorithms::SegmentsReductionKernels::detail {

template< typename Index, typename Lambda, bool AllParameters = CheckFetchLambda< Index, Lambda >::hasAllParameters() >
struct FetchLambdaAdapter
{};

template< typename Index, typename Lambda >
struct FetchLambdaAdapter< Index, Lambda, true >
{
   using ReturnType = decltype( std::declval< Lambda >()( Index(), Index(), Index() ) );

   __cuda_callable__
   static ReturnType
   call( Lambda& f, Index segmentIdx, Index localIdx, Index globalIdx )
   {
      return f( segmentIdx, localIdx, globalIdx );
   }
};

template< typename Index, typename Lambda >
struct FetchLambdaAdapter< Index, Lambda, false >
{
   using ReturnType = decltype( std::declval< Lambda >()( Index() ) );

   __cuda_callable__
   static ReturnType
   call( Lambda& f, Index segmentIdx, Index localIdx, Index globalIdx )
   {
      return f( globalIdx );
   }
};

}  // namespace TNL::Algorithms::SegmentsReductionKernels::detail
