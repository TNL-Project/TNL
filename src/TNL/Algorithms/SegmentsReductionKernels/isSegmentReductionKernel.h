// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::SegmentsReductionKernels {

template< typename SegmentReductionKernel >
struct isSegmentReductionKernel
{
   static constexpr bool value = false;
};

}  // namespace TNL::Algorithms::SegmentsReductionKernels
