// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>
#include <vector>

#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>

/*
 * Helpers for testing multidiagonal matrices of various shapes. The diagonal offsets are not symmetric
 * and the shapes cover matrices with more rows than columns, including empty rows below the lowest
 * diagonal, and matrices with more columns than rows.
 */

inline std::vector< int >
getMultidiagonalTestOffsets()
{
   return { -3, -1, 0, 2 };
}

inline std::vector< std::pair< int, int > >
getMultidiagonalTestShapes()
{
   return { { 5, 5 }, { 6, 4 }, { 4, 6 }, { 8, 3 }, { 1, 1 } };
}

// Returns the column index of the element with the given local index, or -1 if the element does not exist.
inline int
getMultidiagonalColumnIndex( int rowIdx, int localIdx, int columns )
{
   const int columnIdx = rowIdx + getMultidiagonalTestOffsets()[ localIdx ];
   if( columnIdx < 0 || columnIdx >= columns )
      return -1;
   return columnIdx;
}

// Value of the element at the given position used by setupMultidiagonalTestMatrix.
inline int
getMultidiagonalTestValue( int rowIdx, int columnIdx )
{
   return 10 * rowIdx + columnIdx + 1;
}

// Sets the dimensions and the diagonals of the matrix and all its elements to the values given by
// getMultidiagonalTestValue.
template< typename Matrix >
void
setupMultidiagonalTestMatrix( Matrix& matrix, int rows, int columns )
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   const std::vector< int > offsets = getMultidiagonalTestOffsets();
   TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > diagonalOffsets( offsets.size() );
   for( std::size_t i = 0; i < offsets.size(); i++ )
      diagonalOffsets[ i ] = offsets[ i ];
   matrix.setDimensions( rows, columns, diagonalOffsets );
   for( int rowIdx = 0; rowIdx < rows; rowIdx++ )
      for( int localIdx = 0; localIdx < static_cast< int >( offsets.size() ); localIdx++ ) {
         const int columnIdx = getMultidiagonalColumnIndex( rowIdx, localIdx, columns );
         if( columnIdx >= 0 )
            matrix.setElement( rowIdx, columnIdx, static_cast< RealType >( getMultidiagonalTestValue( rowIdx, columnIdx ) ) );
      }
}
