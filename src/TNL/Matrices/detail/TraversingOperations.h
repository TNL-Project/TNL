// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Matrices::detail {

template< typename Matrix >
struct TraversingOperations
{};

}  // namespace TNL::Matrices::detail

#include "TraversingOperationsBase.h"                      // IWYU pragma: keep
#include "TraversingOperations_DenseMatrixView.h"          // IWYU pragma: keep
#include "TraversingOperations_SparseMatrixView.h"         // IWYU pragma: keep
#include "TraversingOperations_TridiagonalMatrixView.h"    // IWYU pragma: keep
#include "TraversingOperations_MultidiagonalMatrixView.h"  // IWYU pragma: keep
#include "TraversingOperations_LambdaMatrix.h"             // IWYU pragma: keep
