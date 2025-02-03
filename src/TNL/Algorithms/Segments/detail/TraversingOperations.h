// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::Segments::detail {

template< typename Segments >
struct TraversingOperations;

}  //namespace TNL::Algorithms::Segments::detail

#include "TraversingOperations_AdaptiveCSR.h"
#include "TraversingOperations_CSR.h"
#include "TraversingOperations_BiEllpack.h"
#include "TraversingOperations_ChunkedEllpack.h"
#include "TraversingOperations_Ellpack.h"
#include "TraversingOperations_SlicedEllpack.h"
