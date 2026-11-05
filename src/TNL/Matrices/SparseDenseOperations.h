// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Matrices {

/**
 * \brief Computes the product of a sparse and a dense matrix, `C = alpha * A * B + beta * C`.
 *
 * The columns of `B` and `C` are processed in tiles of at most `MaxTileSize` columns by
 * \ref TNL::Algorithms::Segments::multiReduceAllSegmentsInTiles over the rows of `A`, so every
 * element of `A` is read once per tile and used for all columns of the tile. It works with all
 * segment types and launch configurations of the sparse matrix and with both row-major and
 * column-major dense matrices. For one column it is equivalent to \ref SparseMatrixBase::vectorProduct.
 *
 * With the default launch configuration, a CSR matrix on GPU and a row-major `B` with more than
 * 4 columns, the product is computed by \ref sparseDenseMatrixProductLanesOverColumns, which
 * is faster for many columns of `B`. The number of lanes and columns per lane are selected by
 * the number of columns of `B` and the precision and `MaxTileSize` is not used then. Otherwise, the default
 * configuration of CSR on GPU is selected by the average number of elements per row: the Tiered
 * mapping for less than 16 elements and the Uniform mapping with 16 threads per row for more.
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

/**
 * \brief Computes the product of a sparse and a dense matrix, `C = alpha * A * B + beta * C`,
 * with the threads of each row of `A` distributed over the columns of `B` and `C`.
 *
 * The product is computed by one \ref TNL::Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions.
 * On GPUs, the columns of `B` and `C` are split into tiles of `Lanes * ColumnsPerLane` columns,
 * a group of `Lanes` threads processes every row of `A` and tile and each thread computes
 * `ColumnsPerLane` consecutive columns of the tile. Hence for row-major `B` the threads of the group
 * read consecutive elements of its rows and no reduction between the threads is needed. All tiles
 * are processed at once, so the product is efficient also for a few rows of `A` and many columns
 * of `B`.
 *
 * The rows of `A` with more than `maxRowLength` elements are split into pieces processed by
 * whole thread blocks, see \ref TNL::Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions.
 *
 * The threads are distributed over the columns only for the CSR format of `A` on GPUs, otherwise
 * the product is computed by the general multireduction, see
 * \ref TNL::Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions.
 * See \ref sparseDenseMatrixProduct for the other parameters.
 *
 * \tparam Lanes The number of threads per row of `A`, a power of two up to the warp size.
 * \tparam ColumnsPerLane The number of consecutive columns computed by each thread.
 * \param maxRowLength The maximal number of elements of the rows processed by one group of
 *        threads, zero means the default.
 */
template<
   int Lanes,
   int ColumnsPerLane,
   typename SparseMatrix,
   typename DenseMatrixB,
   typename DenseMatrixC,
   typename Real = typename SparseMatrix::ComputeRealType >
void
sparseDenseMatrixProductLanesOverColumns(
   const SparseMatrix& A,
   const DenseMatrixB& B,
   DenseMatrixC& C,
   Real alpha = 1,
   Real beta = 0,
   typename SparseMatrix::IndexType maxRowLength = 0 );

}  // namespace TNL::Matrices

#include "SparseDenseOperations.hpp"
