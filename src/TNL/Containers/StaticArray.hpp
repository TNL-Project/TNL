// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeInfo.h>
#include <TNL/TypeTraits.h>
#include <TNL/Math.h>
#include <TNL/Containers/StaticArray.h>

namespace TNL::Containers {

namespace detail {

////
// Static array sort does static loop unrolling of array sort.
// It performs static variant of bubble sort as follows:
//
// for( int k = Size - 1; k > 0; k--)
//   for( int i = 0; i < k; i++ )
//      if( data[ i ] > data[ i+1 ] )
//         swap( data[ i ], data[ i+1 ] );
template< int k, int i, typename Value >
struct StaticArraySort
{
   static constexpr void
   exec( Value* data )
   {
      if( data[ i ] > data[ i + 1 ] )
         swap( data[ i ], data[ i + 1 ] );
      StaticArraySort< k, i + 1, Value >::exec( data );
   }
};

template< int k, typename Value >
struct StaticArraySort< k, k, Value >
{
   static constexpr void
   exec( Value* data )
   {
      StaticArraySort< k - 1, 0, Value >::exec( data );
   }
};

template< typename Value >
struct StaticArraySort< 0, 0, Value >
{
   static constexpr void
   exec( Value* data )
   {}
};

}  // namespace detail

template< int Size, typename Value >
constexpr int
StaticArray< Size, Value >::getSize()
{
   return Size;
}

template< int Size, typename Value >
template< typename T, std::enable_if_t< std::is_same_v< T, int > && ! std::is_same_v< T, Value >, bool > >
__cuda_callable__
constexpr StaticArray< Size, Value >::StaticArray( const T& v )
{
   if constexpr( getSize() > 0 ) {
      for( int i = 0; i < getSize(); i++ )
         data[ i ] = v;
   }
}

template< int Size, typename Value >
template< typename _unused >
constexpr StaticArray< Size, Value >::StaticArray( const Value v[ Size ] )
{
   if constexpr( getSize() > 0 ) {
      for( int i = 0; i < getSize(); i++ )
         data[ i ] = v[ i ];
   }
}

template< int Size, typename Value >
constexpr StaticArray< Size, Value >::StaticArray( const StaticArray& v )
{
   if constexpr( getSize() > 0 ) {
      for( int i = 0; i < getSize(); i++ )
         data[ i ] = v[ i ];
   }
}

template< int Size, typename Value >
template< typename OtherValue >
constexpr StaticArray< Size, Value >::StaticArray( const StaticArray< Size, OtherValue >& v )
{
   if constexpr( getSize() > 0 ) {
      for( int i = 0; i < getSize(); i++ )
         data[ i ] = v[ i ];
   }
}

template< int Size, typename Value >
__cuda_callable__
constexpr StaticArray< Size, Value >::StaticArray( const Value& v )
{
   if constexpr( getSize() > 0 ) {
      for( int i = 0; i < getSize(); i++ )
         data[ i ] = v;
   }
}

template< int Size, typename Value >
template< typename... Values, std::enable_if_t< ( Size > 1 ) && sizeof...( Values ) == Size, bool > >
__cuda_callable__
constexpr StaticArray< Size, Value >::StaticArray( Values&&... values )
: data( { ValueType( std::forward< Values >( values ) )... } )
{}

template< int Size, typename Value >
template< typename OtherValue, std::enable_if_t< std::is_convertible_v< OtherValue, Value >, bool > >
__cuda_callable__
constexpr StaticArray< Size, Value >::StaticArray( const std::initializer_list< OtherValue >& elems )
{
   if constexpr( getSize() > 0 ) {
      const auto* it = elems.begin();
      for( int i = 0; i < getSize() && it != elems.end(); i++ )
         data[ i ] = *it++;
   }
}

template< int Size, typename Value >
template< typename OtherValue >
__cuda_callable__
constexpr StaticArray< Size, Value >::StaticArray( const std::array< OtherValue, Size >& array )
{
   if constexpr( getSize() > 0 ) {
      for( int i = 0; i < getSize(); i++ )
         data[ i ] = array[ i ];
   }
}

template< int Size, typename Value >
__cuda_callable__
constexpr StaticArray< Size, Value >::StaticArray( std::array< Value, Size >&& array )
{
   data = std::move( array );
}

template< int Size, typename Value >
constexpr Value*
StaticArray< Size, Value >::getData() noexcept
{
   return data.data();
}

template< int Size, typename Value >
constexpr const Value*
StaticArray< Size, Value >::getData() const noexcept
{
   return data.data();
}

template< int Size, typename Value >
constexpr const Value&
StaticArray< Size, Value >::operator[]( int i ) const noexcept
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, Size, "Element index is out of bounds." );
   return data[ i ];
}

template< int Size, typename Value >
constexpr Value&
StaticArray< Size, Value >::operator[]( int i ) noexcept
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, Size, "Element index is out of bounds." );
   return data[ i ];
}

template< int Size, typename Value >
constexpr const Value&
StaticArray< Size, Value >::operator()( int i ) const noexcept
{
   return operator[]( i );
}

template< int Size, typename Value >
constexpr Value&
StaticArray< Size, Value >::operator()( int i ) noexcept
{
   return operator[]( i );
}

template< int Size, typename Value >
constexpr Value&
StaticArray< Size, Value >::x() noexcept
{
   return data[ 0 ];
}

template< int Size, typename Value >
constexpr const Value&
StaticArray< Size, Value >::x() const noexcept
{
   return data[ 0 ];
}

template< int Size, typename Value >
constexpr Value&
StaticArray< Size, Value >::y() noexcept
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::y() for arrays with Size < 2." );
   return data[ 1 ];
}

template< int Size, typename Value >
constexpr const Value&
StaticArray< Size, Value >::y() const noexcept
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::y() for arrays with Size < 2." );
   return data[ 1 ];
}

template< int Size, typename Value >
constexpr Value&
StaticArray< Size, Value >::z() noexcept
{
   static_assert( Size > 2, "Cannot call StaticArray< Size, Value >::z() for arrays with Size < 3." );
   return data[ 2 ];
}

template< int Size, typename Value >
constexpr const Value&
StaticArray< Size, Value >::z() const noexcept
{
   static_assert( Size > 2, "Cannot call StaticArray< Size, Value >::z() for arrays with Size < 3." );
   return data[ 2 ];
}

template< int Size, typename Value >
constexpr void
StaticArray< Size, Value >::setElement( IndexType i, ValueType value ) noexcept
{
   this->operator[]( i ) = value;
}

template< int Size, typename Value >
constexpr Value
StaticArray< Size, Value >::getElement( IndexType i ) const noexcept
{
   return this->operator[]( i );
}

template< int Size, typename Value >
constexpr StaticArray< Size, Value >&
StaticArray< Size, Value >::operator=( const StaticArray& v )
{
   if constexpr( getSize() > 0 ) {
      for( int i = 0; i < getSize(); i++ )
         data[ i ] = v[ i ];
   }
   return *this;
}

template< int Size, typename Value >
template< typename T, typename..., typename >
constexpr StaticArray< Size, Value >&
StaticArray< Size, Value >::operator=( const T& v )
{
   if constexpr( getSize() > 0 ) {
      if constexpr( IsStaticArrayType< T >::value ) {
         static_assert( getSize() == T::getSize() );
         for( int i = 0; i < getSize(); i++ )
            data[ i ] = v[ i ];
      }
      else {
         for( int i = 0; i < getSize(); i++ )
            data[ i ] = v;
      }
   }
   return *this;
}

template< int Size, typename Value >
template< typename Array >
constexpr bool
StaticArray< Size, Value >::operator==( const Array& array ) const
{
   if constexpr( getSize() > 0 ) {
      for( int i = 0; i < getSize(); i++ )
         if( ! ( data[ i ] == array[ i ] ) )
            return false;
   }
   return true;
}

template< int Size, typename Value >
template< typename Array >
constexpr bool
StaticArray< Size, Value >::operator!=( const Array& array ) const
{
   return ! this->operator==( array );
}

template< int Size, typename Value >
constexpr void
StaticArray< Size, Value >::setValue( const ValueType& val )
{
   if constexpr( getSize() > 0 ) {
      for( int i = 0; i < getSize(); i++ )
         data[ i ] = val;
   }
}

template< int Size, typename Value >
void
StaticArray< Size, Value >::save( File& file ) const
{
   file.save( getData(), Size );
}

template< int Size, typename Value >
void
StaticArray< Size, Value >::load( File& file )
{
   file.load( getData(), Size );
}

template< int Size, typename Value >
constexpr void
StaticArray< Size, Value >::sort()
{
   detail::StaticArraySort< Size - 1, 0, Value >::exec( getData() );
}

template< int Size, typename Value >
std::ostream&
StaticArray< Size, Value >::write( std::ostream& str, const char* separator ) const
{
   for( int i = 0; i < Size - 1; i++ )
      str << data[ i ] << separator;
   str << data[ Size - 1 ];
   return str;
}

template< int Size, typename Value >
std::ostream&
operator<<( std::ostream& str, const StaticArray< Size, Value >& a )
{
   str << "[ ";
   a.write( str, ", " );
   str << " ]";
   return str;
}

// Serialization of arrays into binary files.
template< int Size, typename Value >
File&
operator<<( File& file, const StaticArray< Size, Value >& array )
{
   for( int i = 0; i < Size; i++ )
      file.save( &array[ i ] );
   return file;
}

template< int Size, typename Value >
File&
operator<<( File&& file, const StaticArray< Size, Value >& array )
{
   File& f = file;
   return f << array;
}

// Deserialization of arrays from binary files.
template< int Size, typename Value >
File&
operator>>( File& file, StaticArray< Size, Value >& array )
{
   for( int i = 0; i < Size; i++ )
      file.load( &array[ i ] );
   return file;
}

template< int Size, typename Value >
File&
operator>>( File&& file, StaticArray< Size, Value >& array )
{
   File& f = file;
   return f >> array;
}

}  // namespace TNL::Containers
