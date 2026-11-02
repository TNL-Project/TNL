// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Matrices {

/**
 * \brief Computes the product of a sparse and a dense matrix, `C = alpha * A * B + beta * C`.
 *
 * The columns of `B` and `C` are processed in tiles of at most `MaxTileSize` columns. Each tile
 * is computed by one \ref TNL::Algorithms::Segments::Segments_multiReduceAllSegments over the rows
 * of `A`, so every element of `A` is read once per tile and used for all columns of the tile.
 * It works with all segment types and launch configurations of the sparse matrix and with both
 * row-major and column-major dense matrices. It is intended for dense matrices with a small
 * number of columns, for one column it is equivalent to \ref SparseMatrixBase::vectorProduct.
 *
 * Symmetric sparse matrices are not supported yet.
 *
 * \tparam MaxTileSize The maximal number of columns of `B` and `C` processed at once.
 * \tparam SparseMatrix The type of the sparse matrix `A` (\ref SparseMatrix or \ref SparseMatrixView).
 * \tparam DenseMatrixB The type of the dense matrix `B` (\ref DenseMatrix or \ref DenseMatrixView).
 * \tparam DenseMatrixC The type of the dense matrix `C` (\ref DenseMatrix or \ref DenseMatrixView).
 * \tparam Real The type of the multiplicators.
 *
 * \param A The sparse matrix with `m` rows and `n` columns.
 * \param B The dense matrix with `n` rows and `k` columns.
 * \param C The dense matrix with `m` rows and `k` columns, the result is stored here.
 * \param alpha The multiplicator of the product `A * B`.
 * \param beta The multiplicator of the original values of `C`. If it is zero, the original
 *             values of `C` are not read.
 * \param launchConfig The launch configuration of the reduction in the segments of `A`.
 */
template<
   int MaxTileSize = 16,
   typename SparseMatrix,
   typename DenseMatrixB,
   typename DenseMatrixC,
   typename Real = typename SparseMatrix::ComputeRealType >
void
sparseDenseMatrixProduct(
   const SparseMatrix& A,
   const DenseMatrixB& B,
   DenseMatrixC& C,
   Real alpha = 1,
   Real beta = 0,
   const Algorithms::Segments::LaunchConfiguration& launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Matrices

#include "SparseDenseOperations.hpp"
