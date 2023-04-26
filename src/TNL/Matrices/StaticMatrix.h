// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <iomanip>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Matrices {

template< typename Value,
          std::size_t Rows,
          std::size_t Columns,
          typename Permutation = std::index_sequence< 0, 1 > >  // identity by default
class StaticMatrix : public Containers::StaticNDArray< Value,
                                                       // note that using std::size_t in SizesHolder does not make sense, since
                                                       // the StaticNDArray is based on StaticArray, which uses int as IndexType
                                                       Containers::SizesHolder< int, Rows, Columns >,
                                                       Permutation >
{
   using Base = Containers::StaticNDArray< Value, Containers::SizesHolder< int, Rows, Columns >, Permutation >;

public:
   // inherit all assignment operators
   using Base::operator=;

   __cuda_callable__
   constexpr StaticMatrix() = default;

   __cuda_callable__
   constexpr StaticMatrix( const StaticMatrix& ) = default;

   __cuda_callable__
   constexpr StaticMatrix( const std::initializer_list< Value >& elems );

   template< typename T >
   __cuda_callable__
   constexpr StaticMatrix( const T& v );

   static constexpr std::size_t
   getRows()
   {
      return Rows;
   }

   __cuda_callable__
   static constexpr std::size_t
   getColumns()
   {
      return Columns;
   }

   template< typename T >
   __cuda_callable__
   constexpr StaticMatrix&
   operator=( const T& v );

   __cuda_callable__
   Containers::StaticVector< Rows, Value >
   operator*( const Containers::StaticVector< Columns, Value >& vector ) const
   {
      Containers::StaticVector< Rows, Value > result;
      for( std::size_t i = 0; i < Rows; i++ ) {
         Value v = 0;
         for( std::size_t j = 0; j < Columns; j++ )
            v += ( *this )( i, j ) * vector[ j ];
         result[ i ] = v;
      }
      return result;
   }

   __cuda_callable__
   constexpr StaticMatrix< Value, Rows, Columns >&
   operator+=( const StaticMatrix< Value, Rows, Columns >& matrix )
   {
      for( std::size_t i = 0; i < Rows; i++ )
         for( std::size_t j = 0; j < Columns; j++ )
            ( *this )( i, j ) += matrix( i, j );

      return *this;
   }

   __cuda_callable__
   constexpr StaticMatrix< Value, Rows, Columns >&
   operator-=( const StaticMatrix< Value, Rows, Columns >& matrix )
   {
      for( std::size_t i = 0; i < Rows; i++ )
         for( std::size_t j = 0; j < Columns; j++ )
            ( *this )( i, j ) -= matrix( i, j );

      return *this;
   }

   template< typename T >
   __cuda_callable__
   constexpr StaticMatrix< Value, Rows, Columns >&
   operator*=( const T& value )
   {
      for( std::size_t i = 0; i < Rows; i++ )
         for( std::size_t j = 0; j < Columns; j++ )
            ( *this )( i, j ) *= value;

      return *this;
   }

   template< typename T >
   __cuda_callable__
   constexpr StaticMatrix< Value, Rows, Columns >&
   operator/=( const T& value )
   {
      for( std::size_t i = 0; i < Rows; i++ )
         for( std::size_t j = 0; j < Columns; j++ )
            ( *this )( i, j ) /= value;

      return *this;
   }

   void
   print( std::ostream& str ) const;
};

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
template< typename T >
__cuda_callable__
constexpr StaticMatrix< Value, Rows, Columns, Permutation >::StaticMatrix( const T& v )
: Containers::StaticNDArray< Value, Containers::SizesHolder< int, Rows, Columns >, Permutation >()
{
   // setValue works but it complaints about calliign __host__ function from __host__ __device__ function.
   this->array = v;
}

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
__cuda_callable__
constexpr StaticMatrix< Value, Rows, Columns, Permutation >::StaticMatrix( const std::initializer_list< Value >& elems )
: Containers::StaticNDArray< Value, Containers::SizesHolder< int, Rows, Columns >, Permutation >()
{
   const auto* it = elems.begin();
   for( std::size_t i = 0; i < ( Rows * Columns ) && it != elems.end(); i++ )
      this->array[ i ] = *it++;
}

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
template< typename T >
__cuda_callable__
constexpr StaticMatrix< Value, Rows, Columns, Permutation >&
StaticMatrix< Value, Rows, Columns, Permutation >::operator=( const T& v )
{
   // setValue works but it complaints about calliign __host__ function from __host__ __device__ function.
   this->array = v;
   return *this;
}

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
std::ostream&
operator<<( std::ostream& str, const StaticMatrix< Value, Rows, Columns, Permutation >& matrix )
{
   matrix.print( str );
   return str;
}

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
__cuda_callable__
StaticMatrix< Value, Rows, Columns, Permutation >
operator+( StaticMatrix< Value, Rows, Columns, Permutation > a, const StaticMatrix< Value, Rows, Columns >& b )
{
   a += b;
   return a;
}

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
__cuda_callable__
StaticMatrix< Value, Rows, Columns, Permutation >
operator-( StaticMatrix< Value, Rows, Columns, Permutation > a, const StaticMatrix< Value, Rows, Columns >& b )
{
   a -= b;
   return a;
}

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation, typename T >
__cuda_callable__
StaticMatrix< Value, Rows, Columns, Permutation >
operator/( StaticMatrix< Value, Rows, Columns, Permutation > a, const T& b )
{
   a /= b;
   return a;
}

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation, typename T >
__cuda_callable__
StaticMatrix< Value, Rows, Columns, Permutation >
operator*( const T& value, StaticMatrix< Value, Rows, Columns, Permutation > a )
{
   a *= value;
   return a;
}

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation, typename T >
__cuda_callable__
StaticMatrix< Value, Rows, Columns, Permutation >
operator*( StaticMatrix< Value, Rows, Columns, Permutation > a, const T& value )
{
   a *= value;
   return a;
}

template< typename Value, std::size_t Rows1, std::size_t SharedDim, std::size_t Columns2, typename Permutation >
StaticMatrix< Value, Rows1, Columns2, Permutation >
operator*( const StaticMatrix< Value, Rows1, SharedDim, Permutation >& matrix1,
           const StaticMatrix< Value, SharedDim, Columns2, Permutation >& matrix2 )
{
   StaticMatrix< Value, Rows1, Columns2, Permutation > result;
   for( std::size_t i = 0; i < Rows1; ++i ) {
      for( std::size_t j = 0; j < Columns2; ++j ) {
         Value value = 0;
         for( std::size_t k = 0; k < SharedDim; ++k ) {
            value += matrix1( i, k ) * matrix2( k, j );
         }
         result( i, j ) = value;
      }
   }

   return result;
}

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
void
StaticMatrix< Value, Rows, Columns, Permutation >::print( std::ostream& str ) const
{
   for( std::size_t row = 0; row < this->getRows(); row++ ) {
      str << "Row: " << row << " -> ";
      for( std::size_t column = 0; column < this->getColumns(); column++ ) {
         std::stringstream str_;
         str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left << ( *this )( row, column );
         str << std::setw( 10 ) << str_.str();
      }
      if( row < this->getRows() - 1 )
         str << std::endl;
   }
}

}  // namespace Matrices
}  // namespace TNL

// Special functions for special cases:
namespace TNL {
namespace Matrices {

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
StaticMatrix< Value, Columns, Rows, Permutation >
transpose( const StaticMatrix< Value, Rows, Columns, Permutation >& A )
{
   StaticMatrix< Value, Columns, Rows, Permutation > result;
   for( std::size_t i = 0; i < Rows; ++i ) {
      for( std::size_t j = 0; j < Columns; ++j ) {
         result( j, i ) = A( i, j );
      }
   }

   return result;
}

template< typename Real >
__cuda_callable__
Real
determinant( const StaticMatrix< Real, 2, 2 >& A )
{
   Real det;
   det = A( 0, 0 ) * A( 1, 1 ) - A( 0, 1 ) * A( 1, 0 );
   return det;
}

template< typename Real >
__cuda_callable__
Real
determinant( const StaticMatrix< Real, 3, 3 >& A )
{
   // clang-format off
   Real det;
   det = A( 0,  0 ) * A( 1, 1 ) * A( 2, 2 ) +
         A( 0,  1 ) * A( 1, 2 ) * A( 2, 0 ) +
         A( 0,  2 ) * A( 1, 0 ) * A( 2, 1 ) -
         A( 2,  0 ) * A( 1, 1 ) * A( 0, 2 ) -
         A( 2,  1 ) * A( 1, 2 ) * A( 0, 0 ) -
         A( 2,  2 ) * A( 1, 0 ) * A( 0, 1 ) ;
   return det;
   // clang-format on
}

template< typename Real >
__cuda_callable__
Real
determinant( const StaticMatrix< Real, 4, 4 >& A )
{
   // clang-format off
   Real det;
   det = A( 0, 3 ) * A( 1, 2 ) * A( 2, 1 ) * A( 3, 0 ) - A( 0, 2 ) * A( 1, 3 ) * A( 2, 1 ) * A( 3, 0 ) -
         A( 0, 3 ) * A( 1, 1 ) * A( 2, 2 ) * A( 3, 0 ) + A( 0, 1 ) * A( 1, 3 ) * A( 2, 2 ) * A( 3, 0 ) +
         A( 0, 2 ) * A( 1, 1 ) * A( 2, 3 ) * A( 3, 0 ) - A( 0, 1 ) * A( 1, 2 ) * A( 2, 3 ) * A( 3, 0 ) -
         A( 0, 3 ) * A( 1, 2 ) * A( 2, 0 ) * A( 3, 1 ) + A( 0, 2 ) * A( 1, 3 ) * A( 2, 0 ) * A( 3, 1 ) +
         A( 0, 3 ) * A( 1, 0 ) * A( 2, 2 ) * A( 3, 1 ) - A( 0, 0 ) * A( 1, 3 ) * A( 2, 2 ) * A( 3, 1 ) -
         A( 0, 2 ) * A( 1, 0 ) * A( 2, 3 ) * A( 3, 1 ) + A( 0, 0 ) * A( 1, 2 ) * A( 2, 3 ) * A( 3, 1 ) +
         A( 0, 3 ) * A( 1, 1 ) * A( 2, 0 ) * A( 3, 2 ) - A( 0, 1 ) * A( 1, 3 ) * A( 2, 0 ) * A( 3, 2 ) -
         A( 0, 3 ) * A( 1, 0 ) * A( 2, 1 ) * A( 3, 2 ) + A( 0, 0 ) * A( 1, 3 ) * A( 2, 1 ) * A( 3, 2 ) +
         A( 0, 1 ) * A( 1, 0 ) * A( 2, 3 ) * A( 3, 2 ) - A( 0, 0 ) * A( 1, 1 ) * A( 2, 3 ) * A( 3, 2 ) -
         A( 0, 2 ) * A( 1, 1 ) * A( 2, 0 ) * A( 3, 3 ) + A( 0, 1 ) * A( 1, 2 ) * A( 2, 0 ) * A( 3, 3 ) +
         A( 0, 2 ) * A( 1, 0 ) * A( 2, 1 ) * A( 3, 3 ) - A( 0, 0 ) * A( 1, 2 ) * A( 2, 1 ) * A( 3, 3 ) -
         A( 0, 1 ) * A( 1, 0 ) * A( 2, 2 ) * A( 3, 3 ) + A( 0, 0 ) * A( 1, 1 ) * A( 2, 2 ) * A( 3, 3 ) ;
   return det;
   // clang-format on
}

template< typename Real >
__cuda_callable__
StaticMatrix< Real, 2, 2 >
inverse( const StaticMatrix< Real, 2, 2 >& A )
{
   Real det = determinant( A );
   StaticMatrix< Real, 2, 2 > invA;

   // clang-format off
   invA( 0, 0 ) =   A( 1, 1 );
   invA( 0, 1 ) = - A( 0, 1 );
   invA( 1, 0 ) = - A( 1, 0 );
   invA( 1, 1 ) =   A( 0, 0 );
   // clang-format on

   return invA / det;
}

template< typename Real >
__cuda_callable__
StaticMatrix< Real, 3, 3 >
inverse( const StaticMatrix< Real, 3, 3 >& A )
{
   Real det = determinant( A );
   StaticMatrix< Real, 3, 3 > invA;

   // clang-format off
   invA( 0, 0 ) =    ( A( 1, 1 ) * A( 2, 2 ) - A( 1, 2 ) * A( 2, 1 ) ),
   invA( 0, 1 ) =  - ( A( 0, 1 ) * A( 2, 2 ) - A( 0, 2 ) * A( 2, 1 ) ),
   invA( 0, 2 ) =    ( A( 0, 1 ) * A( 1, 2 ) - A( 0, 2 ) * A( 1, 1 ) ),
   invA( 1, 0 ) =  - ( A( 1, 0 ) * A( 2, 2 ) - A( 1, 2 ) * A( 2, 0 ) ),
   invA( 1, 1 ) =    ( A( 0, 0 ) * A( 2, 2 ) - A( 0, 2 ) * A( 2, 0 ) ),
   invA( 1, 2 ) =  - ( A( 0, 0 ) * A( 1, 2 ) - A( 0, 2 ) * A( 1, 0 ) ),
   invA( 2, 0 ) =    ( A( 1, 0 ) * A( 2, 1 ) - A( 1, 1 ) * A( 2, 0 ) ),
   invA( 2, 1 ) =  - ( A( 0, 0 ) * A( 2, 1 ) - A( 0, 1 ) * A( 2, 0 ) ),
   invA( 2, 2 ) =    ( A( 0, 0 ) * A( 1, 1 ) - A( 0, 1 ) * A( 1, 0 ) );
   // clang-format on

   return invA / det;
}

template< typename Real >
__cuda_callable__
StaticMatrix< Real, 4, 4 >
inverse( const StaticMatrix< Real, 4, 4 >& A )
{
   Real det = determinant( A );
   StaticMatrix< Real, 4, 4 > invA;

   // clang-format off
   invA( 0, 0 ) = ( A( 1, 1 ) * ( A( 2, 2 ) * A( 3, 3 ) - A( 2, 3 ) * A( 3, 2 ) ) +
                    A( 1, 2 ) * ( A( 2, 3 ) * A( 3, 1 ) - A( 2, 1 ) * A( 3, 3 ) ) +
                    A( 1, 3 ) * ( A( 2, 1 ) * A( 3, 2 ) - A( 2, 2 ) * A( 3, 1 ) ) );
   invA( 1, 0 ) = ( A( 1, 0 ) * ( A( 2, 3 ) * A( 3, 2 ) - A( 2, 2 ) * A( 3, 3 ) ) +
                    A( 1, 2 ) * ( A( 2, 0 ) * A( 3, 3 ) - A( 2, 3 ) * A( 3, 0 ) ) +
                    A( 1, 3 ) * ( A( 2, 2 ) * A( 3, 0 ) - A( 2, 0 ) * A( 3, 2 ) ) );
   invA( 2, 0 ) = ( A( 1, 0 ) * ( A( 2, 1 ) * A( 3, 3 ) - A( 2, 3 ) * A( 3, 1 ) ) +
                    A( 1, 1 ) * ( A( 2, 3 ) * A( 3, 0 ) - A( 2, 0 ) * A( 3, 3 ) ) +
                    A( 1, 3 ) * ( A( 2, 0 ) * A( 3, 1 ) - A( 2, 1 ) * A( 3, 0 ) ) );
   invA( 3, 0 ) = ( A( 1, 0 ) * ( A( 2, 2 ) * A( 3, 1 ) - A( 2, 1 ) * A( 3, 2 ) ) +
                    A( 1, 1 ) * ( A( 2, 0 ) * A( 3, 2 ) - A( 2, 2 ) * A( 3, 0 ) ) +
                    A( 1, 2 ) * ( A( 2, 1 ) * A( 3, 0 ) - A( 2, 0 ) * A( 3, 1 ) ) );
   invA( 0, 1 ) = ( A( 0, 1 ) * ( A( 2, 3 ) * A( 3, 2 ) - A( 2, 2 ) * A( 3, 3 ) ) +
                    A( 0, 2 ) * ( A( 2, 1 ) * A( 3, 3 ) - A( 2, 3 ) * A( 3, 1 ) ) +
                    A( 0, 3 ) * ( A( 2, 2 ) * A( 3, 1 ) - A( 2, 1 ) * A( 3, 2 ) ) );
   invA( 1, 1 ) = ( A( 0, 0 ) * ( A( 2, 2 ) * A( 3, 3 ) - A( 2, 3 ) * A( 3, 2 ) ) +
                    A( 0, 2 ) * ( A( 2, 3 ) * A( 3, 0 ) - A( 2, 0 ) * A( 3, 3 ) ) +
                    A( 0, 3 ) * ( A( 2, 0 ) * A( 3, 2 ) - A( 2, 2 ) * A( 3, 0 ) ) );
   invA( 2, 1 ) = ( A( 0, 0 ) * ( A( 2, 3 ) * A( 3, 1 ) - A( 2, 1 ) * A( 3, 3 ) ) +
                    A( 0, 1 ) * ( A( 2, 0 ) * A( 3, 3 ) - A( 2, 3 ) * A( 3, 0 ) ) +
                    A( 0, 3 ) * ( A( 2, 1 ) * A( 3, 0 ) - A( 2, 0 ) * A( 3, 1 ) ) );
   invA( 3, 1 ) = ( A( 0, 0 ) * ( A( 2, 1 ) * A( 3, 2 ) - A( 2, 2 ) * A( 3, 1 ) ) +
                    A( 0, 1 ) * ( A( 2, 2 ) * A( 3, 0 ) - A( 2, 0 ) * A( 3, 2 ) ) +
                    A( 0, 2 ) * ( A( 2, 0 ) * A( 3, 1 ) - A( 2, 1 ) * A( 3, 0 ) ) );
   invA( 0, 2 ) = ( A( 0, 1 ) * ( A( 1, 2 ) * A( 3, 3 ) - A( 1, 3 ) * A( 3, 2 ) ) +
                    A( 0, 2 ) * ( A( 1, 3 ) * A( 3, 1 ) - A( 1, 1 ) * A( 3, 3 ) ) +
                    A( 0, 3 ) * ( A( 1, 1 ) * A( 3, 2 ) - A( 1, 2 ) * A( 3, 1 ) ) );
   invA( 1, 2 ) = ( A( 0, 0 ) * ( A( 1, 3 ) * A( 3, 2 ) - A( 1, 2 ) * A( 3, 3 ) ) +
                    A( 0, 2 ) * ( A( 1, 0 ) * A( 3, 3 ) - A( 1, 3 ) * A( 3, 0 ) ) +
                    A( 0, 3 ) * ( A( 1, 2 ) * A( 3, 0 ) - A( 1, 0 ) * A( 3, 2 ) ) );
   invA( 2, 2 ) = ( A( 0, 0 ) * ( A( 1, 1 ) * A( 3, 3 ) - A( 1, 3 ) * A( 3, 1 ) ) +
                    A( 0, 1 ) * ( A( 1, 3 ) * A( 3, 0 ) - A( 1, 0 ) * A( 3, 3 ) ) +
                    A( 0, 3 ) * ( A( 1, 0 ) * A( 3, 1 ) - A( 1, 1 ) * A( 3, 0 ) ) );
   invA( 3, 2 ) = ( A( 0, 0 ) * ( A( 1, 2 ) * A( 3, 1 ) - A( 1, 1 ) * A( 3, 2 ) ) +
                    A( 0, 1 ) * ( A( 1, 0 ) * A( 3, 2 ) - A( 1, 2 ) * A( 3, 0 ) ) +
                    A( 0, 2 ) * ( A( 1, 1 ) * A( 3, 0 ) - A( 1, 0 ) * A( 3, 1 ) ) );
   invA( 0, 3 ) = ( A( 0, 1 ) * ( A( 1, 3 ) * A( 2, 2 ) - A( 1, 2 ) * A( 2, 3 ) ) +
                    A( 0, 2 ) * ( A( 1, 1 ) * A( 2, 3 ) - A( 1, 3 ) * A( 2, 1 ) ) +
                    A( 0, 3 ) * ( A( 1, 2 ) * A( 2, 1 ) - A( 1, 1 ) * A( 2, 2 ) ) );
   invA( 1, 3 ) = ( A( 0, 0 ) * ( A( 1, 2 ) * A( 2, 3 ) - A( 1, 3 ) * A( 2, 2 ) ) +
                    A( 0, 2 ) * ( A( 1, 3 ) * A( 2, 0 ) - A( 1, 0 ) * A( 2, 3 ) ) +
                    A( 0, 3 ) * ( A( 1, 0 ) * A( 2, 2 ) - A( 1, 2 ) * A( 2, 0 ) ) );
   invA( 2, 3 ) = ( A( 0, 0 ) * ( A( 1, 3 ) * A( 2, 1 ) - A( 1, 1 ) * A( 2, 3 ) ) +
                    A( 0, 1 ) * ( A( 1, 0 ) * A( 2, 3 ) - A( 1, 3 ) * A( 2, 0 ) ) +
                    A( 0, 3 ) * ( A( 1, 1 ) * A( 2, 0 ) - A( 1, 0 ) * A( 2, 1 ) ) );
   invA( 3, 3 ) = ( A( 0, 0 ) * ( A( 1, 1 ) * A( 2, 2 ) - A( 1, 2 ) * A( 2, 1 ) ) +
                    A( 0, 1 ) * ( A( 1, 2 ) * A( 2, 0 ) - A( 1, 0 ) * A( 2, 2 ) ) +
                    A( 0, 2 ) * ( A( 1, 0 ) * A( 2, 1 ) - A( 1, 1 ) * A( 2, 0 ) ) );
   // clang-format on

   return invA / det;
}

template< typename Real >
__cuda_callable__
Containers::StaticVector< 2, Real >
solve( const StaticMatrix< Real, 2, 2 >& A, const Containers::StaticVector< 2, Real >& b )
{
   return inverse( A ) * b;
}

template< typename Real >
__cuda_callable__
Containers::StaticVector< 3, Real >
solve( const StaticMatrix< Real, 3, 3 >& A, const Containers::StaticVector< 3, Real >& b )
{
   return inverse( A ) * b;
}

template< typename Real >
__cuda_callable__
Containers::StaticVector< 4, Real >
solve( const StaticMatrix< Real, 4, 4 >& A, const Containers::StaticVector< 4, Real >& b )
{
   return inverse( A ) * b;
}

}  // namespace Matrices
}  // namespace TNL
