// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace TNL::Matrices {

enum class TransposeState : std::uint8_t
{
   None,
   Transpose
};

template< typename ResultMatrix, typename Matrix1, typename Matrix2, typename Real, int tileDim = 16 >
void
getMatrixProduct( ResultMatrix& resultMatrix,
                  const Matrix1& matrix1,
                  const Matrix2& matrix2,
                  Real matrixMultiplicator = 1.0,
                  TransposeState transposeA = TransposeState::None,
                  TransposeState transposeB = TransposeState::None );

template< typename ResultMatrix, typename Matrix, typename Real, int tileDim = 16 >
void
getTransposition( ResultMatrix& resultMatrix, const Matrix& matrix, Real matrixMultiplicator = 1.0 );

template< typename Matrix, typename Real, int tileDim = 16 >
void
getInPlaceTransposition( Matrix& matrix, Real matrixMultiplicator = 1.0 );

}  // namespace TNL::Matrices

#include "DenseOperations.hpp"
