// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "MultiReducingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

// The general implementation for the segment types without a specialization.
template< typename Segments >
struct MultiReducingOperations : public MultiReducingOperationsBase< Segments >
{};

}  // namespace TNL::Algorithms::Segments::detail

#include "MultiReducingOperations_CSR.h"
