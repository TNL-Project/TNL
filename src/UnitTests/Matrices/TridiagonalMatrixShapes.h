// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>
#include <vector>

/*
 * Helpers for testing tridiagonal matrices of various shapes. The shapes cover the first row,
 * the last row of square matrices, the row having only the subdiagonal element in matrices with
 * more rows than columns, the empty rows beyond it and matrices with a single column.
 */

inline std::vector< std::pair< int, int > >
getTridiagonalTestShapes()
{
   return { { 4, 4 }, { 3, 5 }, { 5, 4 }, { 6, 3 }, { 1, 1 }, { 3, 1 } };
}

// Returns the column index of the element with the given local index, or -1 if the element does not exist.
inline int
getTridiagonalColumnIndex( int rowIdx, int localIdx, int columns )
{
   const int columnIdx = rowIdx + localIdx - 1;
   if( columnIdx < 0 || columnIdx >= columns )
      return -1;
   return columnIdx;
}

// Value of the element at the given position used by setupTridiagonalTestMatrix.
inline int
getTridiagonalTestValue( int rowIdx, int columnIdx )
{
   return 10 * rowIdx + columnIdx + 1;
}

// Sets all elements of the tridiagonal pattern to the values given by getTridiagonalTestValue.
template< typename Matrix >
void
setupTridiagonalTestMatrix( Matrix& matrix )
{
   using RealType = typename Matrix::RealType;
   for( int rowIdx = 0; rowIdx < matrix.getRows(); rowIdx++ )
      for( int localIdx = 0; localIdx < 3; localIdx++ ) {
         const int columnIdx = getTridiagonalColumnIndex( rowIdx, localIdx, matrix.getColumns() );
         if( columnIdx >= 0 )
            matrix.setElement( rowIdx, columnIdx, static_cast< RealType >( getTridiagonalTestValue( rowIdx, columnIdx ) ) );
      }
}
