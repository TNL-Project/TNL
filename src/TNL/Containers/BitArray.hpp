// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

#include <TNL/Containers/BitArray.h>

namespace TNL {
namespace Containers {

template< int Size, typename Base >
constexpr int
BitArray< Size, Base >::getSize()
{
   return Size;
}

template< int Size, typename Base >
constexpr int
BitArray< Size, Base >::getBitsPerElement()
{
   return sizeof( BaseType ) * 8;
}

template< int Size, typename Base >
constexpr int
BitArray< Size, Base >::getDataSize()
{
   return Size / getBitsPerElement() + ( getSize() % getBitsPerElement() != 0 );
}

template< int Size, typename Base >
__cuda_callable__
constexpr BitArray< Size, Base >::BitArray()
{
   for( int i = 0; i < getDataSize(); i++ )
      data[ i ] = (BaseType) 0;
}

template< int Size, typename Base >
__cuda_callable__
constexpr BitArray< Size, Base >::BitArray( const BaseType* b, int b_size )
{
   int i = 0;
   while( i < b_size && i < getDataSize() ) {
      data[ i ] = b[ getDataSize() - i - 1 ];
      i++;
   }
   while( i < getDataSize() ) {
      data[ i++ ] = (BaseType) 0;
   }
}

template< int Size, typename Base >
__cuda_callable__
constexpr BitArray< Size, Base >::BitArray( const BaseType& b )
{
   data[ 0 ] = b;
   int i = 1;
   while( i < getDataSize() )
      data[ i++ ] = (BaseType) 0;
}

template< int Size, typename Base >
__cuda_callable__
constexpr BitArray< Size, Base >::BitArray( const BitArray& array )
{
   for( int i = 0; i < getDataSize(); i++ )
      data[ i ] = array.data[ i ];
}

template< int Size, typename Base >
__cuda_callable__
constexpr BitArray< Size, Base >&
BitArray< Size, Base >::operator=( const BitArray& b )
{
   for( int i = 0; i < getDataSize(); i++ )
      data[ i ] = b.data[ i ];
   return *this;
}

template< int Size, typename Base >
__cuda_callable__
constexpr bool
BitArray< Size, Base >::operator==( const BitArray& b ) const
{
   for( int i = 0; i < getDataSize(); i++ )
      if( data[ i ] != b.data[ i ] )
         return false;
   return true;
}

template< int Size, typename Base >
   template< typename BaseType_ >
__cuda_callable__
constexpr bool
BitArray< Size, Base >::operator==( const BitArray< Size, BaseType_ >& b ) const
{
   for( int i = 0; i < getSize(); i++ )
      if( ( *this )[ i ] != b[ i ] )
         return false;
   return true;
}

template< int Size, typename Base >
__cuda_callable__
constexpr bool
BitArray< Size, Base >::operator!=( const BitArray& b ) const
{
   return ! this->operator==( b );
}

template< int Size, typename Base >
template< typename BaseType_ >
__cuda_callable__
constexpr bool
BitArray< Size, Base >::operator!=( const BitArray< Size, BaseType_ >& b ) const
{
   return ! this->operator==( b );
}

template< int Size, typename Base >
__cuda_callable__
constexpr Base
BitArray< Size, Base >::operator&( const BaseType& b ) const
{
   return data[ 0 ] & b;
}

template< int Size, typename Base >
__cuda_callable__
constexpr BitArray< Size, Base >&
BitArray< Size, Base >::operator++( int )
{
   int i = 0;
   while( data[ i ] == std::numeric_limits< BaseType >::max() && i < getDataSize() )
      data[ i++ ] = (BaseType) 0;
   data[ i ]++;
   return *this;
}

template< int Size, typename Base >
__cuda_callable__
constexpr BitArray< Size, Base >&
BitArray< Size, Base >::operator>>=( int i )
{
   while( i-- > 0 )
      shiftRight();
   return *this;
}

template< int Size, typename Base >
__cuda_callable__
constexpr bool
BitArray< Size, Base >::operator[]( int i ) const
{
   int idx = 0;
   while( i >= getBitsPerElement() ) {
      idx++;
      i -= getBitsPerElement();
   }
   BaseType mask = ( 1 << i );
   return data[ idx ] & mask;
}

template< int Size, typename Base >
__cuda_callable__
constexpr void
BitArray< Size, Base >::reset()
{
   for( int i = 0; i < getDataSize(); i++ )
      data[ i ] = (BaseType) 0;
}

template< int Size, typename Base >
__cuda_callable__
constexpr void
BitArray< Size, Base >::shiftRight()
{
   constexpr BaseType highestBit = (BaseType) 1 << ( getBitsPerElement() - 1 );
   for( int i = 0; i < getDataSize() - 1; i++ ) {
      data[ i ] >>= 1;
      if( data[ i + 1 ] & 1 )
         data[ i ] |= highestBit;
   }
   data[ getDataSize() - 1 ] >>= 1;
}

template< int Size, typename Base >
std::ostream&
operator<<( std::ostream& str, const BitArray< Size, Base >& b )
{
   for( int i = Size - 1; i >= 0; i-- ) {
      if( b[ i ] )
         str << 1;
      else
         str << 0;
   }
   return str;
}

template< int Size, typename Base >
std::ostream&
BitArray< Size, Base >::print( std::ostream& str ) const
{
   std::cout << " Bits per element: " << getBitsPerElement() << std::endl;
   std::cout << " Data size: " << getDataSize() << std::endl;
   for( int i = getDataSize() - 1; i >= 0; i-- )
      str << (long long int) data[ i ] << ",";
   return str;
}
}  // namespace Containers
}  // namespace TNL
