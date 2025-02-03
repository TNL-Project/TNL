// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::Segments::detail {

template< typename Segments >
struct ReducingOperations;

}  //namespace TNL::Algorithms::Segments::detail

#include "ReducingOperations_AdaptiveCSR.h"
#include "ReducingOperations_BiEllpack.h"
#include "ReducingOperations_ChunkedEllpack.h"
#include "ReducingOperations_CSR.h"
#include "ReducingOperations_Ellpack.h"
#include "ReducingOperations_SlicedEllpack.h"
