// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>

#include "../Segments/detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename Kernel >
struct isSegmentsReductionKernel : public std::false_type
{};

template< typename Kernel >
inline constexpr bool isSegmentsReductionKernel_v = isSegmentsReductionKernel< Kernel >::value;

}  //  namespace TNL::Algorithms::SegmentsReductionKernels
