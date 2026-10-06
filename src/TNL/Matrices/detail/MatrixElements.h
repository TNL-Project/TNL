// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace TNL::Matrices::detail {

// Compares matrix elements given as tuples ( row, column, value ) by their position in the row-major order.
struct MatrixElementsPositionLess
{
   template< typename Element >
   bool
   operator()( const Element& a, const Element& b ) const
   {
      return std::get< 0 >( a ) < std::get< 0 >( b )
          || ( std::get< 0 >( a ) == std::get< 0 >( b ) && std::get< 1 >( a ) < std::get< 1 >( b ) );
   }
};

// Sorts the matrix elements in the row-major order and checks that each position appears only once.
template< typename Index, typename Real >
void
sortMatrixElements( std::vector< std::tuple< Index, Index, Real > >& elements )
{
   const MatrixElementsPositionLess less;
   if( ! std::is_sorted( elements.begin(), elements.end(), less ) )
      std::sort( elements.begin(), elements.end(), less );
   const auto duplicate = std::adjacent_find(
      elements.begin(),
      elements.end(),
      [ & ]( const auto& a, const auto& b )
      {
         return ! less( a, b );
      } );
   if( duplicate != elements.end() )
      throw std::logic_error(
         "The matrix element at position (" + std::to_string( std::get< 0 >( *duplicate ) ) + ", "
         + std::to_string( std::get< 1 >( *duplicate ) ) + ") appears more than once in the input data." );
}

// Returns a pointer to the value of the matrix element at the given position in the sorted elements, or nullptr if there is
// no such element.
template< typename Index, typename Real >
const Real*
findMatrixElement( const std::vector< std::tuple< Index, Index, Real > >& sortedElements, Index row, Index column )
{
   const std::tuple< Index, Index, Real > key( row, column, Real{} );
   const auto element = std::lower_bound( sortedElements.begin(), sortedElements.end(), key, MatrixElementsPositionLess{} );
   if( element == sortedElements.end() || std::get< 0 >( *element ) != row || std::get< 1 >( *element ) != column )
      return nullptr;
   return &std::get< 2 >( *element );
}

}  // namespace TNL::Matrices::detail
