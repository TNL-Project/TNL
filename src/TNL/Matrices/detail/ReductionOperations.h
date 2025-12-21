// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Matrices::detail {

template< typename Matrix >
struct ReductionOperations
{};

}  //namespace TNL::Matrices::detail

#include "ReductionOperations_DenseMatrixView.h"
#include "ReductionOperations_SparseMatrixView.h"
