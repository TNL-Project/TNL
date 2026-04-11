// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/AdaptiveCSRView.h>
#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include "TraversingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
struct TraversingOperations< AdaptiveCSRView< Device, Index > > : public TraversingOperations< CSRView< Device, Index > >
{};

}  // namespace TNL::Algorithms::Segments::detail
