// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Matrices::detail {

template< typename Matrix >
struct ReductionOperations
{};

}  // namespace TNL::Matrices::detail

#include "ReductionOperationsBase.h"                      // IWYU pragma: keep
#include "ReductionOperations_DenseMatrixView.h"          // IWYU pragma: keep
#include "ReductionOperations_SparseMatrixView.h"         // IWYU pragma: keep
#include "ReductionOperations_TridiagonalMatrixView.h"    // IWYU pragma: keep
#include "ReductionOperations_MultidiagonalMatrixView.h"  // IWYU pragma: keep
#include "ReductionOperations_LambdaMatrix.h"             // IWYU pragma: keep
