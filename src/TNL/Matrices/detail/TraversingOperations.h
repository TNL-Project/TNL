// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Matrices::detail {

template< typename Matrix >
struct TraversingOperations
{};

}  //namespace TNL::Matrices::detail

#include "TraversingOperations_DenseMatrixView.h"
#include "TraversingOperations_SparseMatrixView.h"
