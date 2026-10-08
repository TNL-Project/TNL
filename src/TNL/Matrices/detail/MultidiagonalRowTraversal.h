// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

namespace TNL::Matrices::detail {

/**
 * \brief Calls `function( localIdx, columnIdx, globalIdx )` for each element of the given row of a multidiagonal matrix.
 *
 * The local index `localIdx` is the index of the diagonal in the array of the diagonals offsets. The column index
 * is `rowIdx + diagonalOffsets[ localIdx ]` and `globalIdx` is the position of the element in the array of values.
 * Elements whose column index lies outside the matrix are skipped. This covers the padding elements in the first
 * and the last rows and the rows of matrices with more rows than columns that lie below the lowest diagonal and
 * are empty.
 *
 * \param indexer The indexer of the multidiagonal matrix, i.e. `MultidiagonalMatrixIndexer`.
 * \param diagonalOffsets The view of the diagonals offsets stored on the matrix device.
 * \param rowIdx The index of the matrix row.
 * \param function The function called for each element of the row.
 */
template< typename Indexer, typename DiagonalOffsetsView, typename RowIndex, typename Function >
__cuda_callable__
void
forMultidiagonalRowElements(
   const Indexer& indexer,
   const DiagonalOffsetsView& diagonalOffsets,
   RowIndex rowIdx,
   Function&& function )
{
   using IndexType = typename Indexer::IndexType;
   const IndexType row = rowIdx;
   for( IndexType localIdx = 0; localIdx < indexer.getDiagonals(); localIdx++ ) {
      const IndexType columnIdx = row + diagonalOffsets[ localIdx ];
      if( columnIdx >= 0 && columnIdx < indexer.getColumns() )
         function( localIdx, columnIdx, indexer.getGlobalIndex( row, localIdx ) );
   }
}

}  // namespace TNL::Matrices::detail
