// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Matrices/MatrixBase.h>

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

// Thrown by sortMatrixElements when a position appears more than once and the duplicates are not allowed.
template< typename Index >
class DuplicateMatrixElementError : public std::logic_error
{
public:
   DuplicateMatrixElementError( Index row, Index column )
   : std::logic_error(
        "The matrix element at position (" + std::to_string( row ) + ", " + std::to_string( column )
        + ") appears more than once in the input data." ),
     row( row ),
     column( column )
   {}

   [[nodiscard]] Index
   getRow() const
   {
      return row;
   }

   [[nodiscard]] Index
   getColumn() const
   {
      return column;
   }

private:
   Index row;
   Index column;
};

// Sorts the matrix elements in the row-major order and resolves the elements at the same position.
template< typename Index, typename Real >
void
sortMatrixElements( std::vector< std::tuple< Index, Index, Real > >& elements, MatrixElementsDuplicates duplicates )
{
   const MatrixElementsPositionLess less;
   if( ! std::is_sorted( elements.begin(), elements.end(), less ) ) {
      // The stable sort keeps the elements at the same position in the order of the input data, which is needed
      // for KeepLast and which makes the results of Sum and Average independent of the sorting algorithm.
      if( duplicates == MatrixElementsDuplicates::Error )
         std::sort( elements.begin(), elements.end(), less );
      else
         std::stable_sort( elements.begin(), elements.end(), less );
   }

   if( duplicates == MatrixElementsDuplicates::Error ) {
      const auto duplicate = std::adjacent_find(
         elements.begin(),
         elements.end(),
         [ & ]( const auto& a, const auto& b )
         {
            return ! less( a, b );
         } );
      if( duplicate != elements.end() )
         throw DuplicateMatrixElementError< Index >( std::get< 0 >( *duplicate ), std::get< 1 >( *duplicate ) );
      return;
   }

   if( elements.empty() )
      return;
   // Divides the sum of the elements at the same position by their count.
   auto average = [ & ]( Real& value, std::size_t count )
   {
      if constexpr( ! std::is_same_v< Real, bool > )
         if( duplicates == MatrixElementsDuplicates::Average && count > 1 )
            value /= static_cast< Real >( count );
   };
   std::size_t last = 0;
   std::size_t count = 1;
   for( std::size_t i = 1; i < elements.size(); i++ ) {
      Real& value = std::get< 2 >( elements[ last ] );
      if( less( elements[ last ], elements[ i ] ) ) {
         average( value, count );
         elements[ ++last ] = std::move( elements[ i ] );
         count = 1;
         continue;
      }
      count++;
      if( duplicates == MatrixElementsDuplicates::KeepLast )
         value = std::get< 2 >( elements[ i ] );
      else if constexpr( std::is_same_v< Real, bool > )
         value = value || std::get< 2 >( elements[ i ] );
      else
         value += std::get< 2 >( elements[ i ] );
   }
   average( std::get< 2 >( elements[ last ] ), count );
   elements.resize( last + 1 );
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
