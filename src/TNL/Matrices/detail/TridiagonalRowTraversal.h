// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

namespace TNL::Matrices::detail {

/**
 * \brief Calls `function( localIdx, columnIdx, globalIdx )` for each element of the given row of a tridiagonal matrix.
 *
 * The local index `localIdx` is the index of the diagonal: 0 for the subdiagonal, 1 for the main diagonal
 * and 2 for the superdiagonal. The column index is `rowIdx + localIdx - 1` and `globalIdx` is the position
 * of the element in the array of values. Elements whose column index lies outside the matrix are skipped.
 * This covers the first row, the last row, the matrices having a single column and the rows of matrices
 * with more rows than columns that lie beyond the last column and are empty.
 *
 * \param indexer The indexer of the tridiagonal matrix, i.e. `TridiagonalMatrixIndexer`.
 * \param rowIdx The index of the matrix row.
 * \param function The function called for each element of the row.
 */
template< typename Indexer, typename RowIndex, typename Function >
__cuda_callable__
void
forTridiagonalRowElements( const Indexer& indexer, RowIndex rowIdx, Function&& function )
{
   using IndexType = typename Indexer::IndexType;
   const IndexType row = rowIdx;
   for( IndexType localIdx = 0; localIdx < 3; localIdx++ ) {
      const IndexType columnIdx = row + localIdx - 1;
      if( columnIdx >= 0 && columnIdx < indexer.getColumns() )
         function( localIdx, columnIdx, indexer.getGlobalIndex( row, localIdx ) );
   }
}

}  // namespace TNL::Matrices::detail
