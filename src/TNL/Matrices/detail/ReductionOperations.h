// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Matrices::detail {

template< typename Matrix >
struct ReductionOperations
{};

}  // namespace TNL::Matrices::detail

#include "ReductionOperationsBase.h"
#include "ReductionOperations_DenseMatrixView.h"
#include "ReductionOperations_SparseMatrixView.h"
#include "ReductionOperations_TridiagonalMatrixView.h"
#include "ReductionOperations_MultidiagonalMatrixView.h"
#include "ReductionOperations_LambdaMatrix.h"
