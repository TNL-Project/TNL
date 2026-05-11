// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace TNL::Matrices {

template< typename Matrix1, typename Matrix2 >
void
copySparseToDenseMatrix( Matrix1& A, const Matrix2& B );

template< typename Matrix1, typename Matrix2 >
void
copyDenseToSparseMatrix( Matrix1& A, const Matrix2& B );

}  // namespace TNL::Matrices

#include "DenseSparseOperations.hpp"
