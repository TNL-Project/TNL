// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <TNL/File.h>

namespace TNL::Containers {

/**
 * \brief Array with constant size.
 *
 * \tparam Size Size of static array. Number of its elements.
 * \tparam Value Type of the values in static array.
 */
template< int Size, typename Value >
class StaticArray
{
public:
   /**
    * \brief Type of elements stored in this array.
    */
   using ValueType = Value;

   /**
    * \brief Type being used for the array elements indexing.
    */
   using IndexType = int;

   /**
    * \brief Gets size of this array.
    */
   [[nodiscard]] static constexpr int
   getSize();

   /**
    * \brief Default constructor.
    */
   __cuda_callable__
   constexpr StaticArray() = default;

   /**
    * \brief Constructor from static array.
    *
    * \param v Input array.
    */
   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   constexpr StaticArray( const Value v[ Size ] );

   /**
    * \brief Constructor that sets all array components to value \e v.
    *
    * \param v Reference to a value.
    */
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   constexpr StaticArray( const Value& v );

   /**
    * \brief Copy constructor.
    *
    * Constructs a copy of another static array \e v.
    */
   // NOTE: as of nvcc 11.8, the default/implicit copy-constructor for std::array (which is based on aggregate initialization)
   // does not work correctly in device code, so we must defaine our own copy-constructor
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   constexpr StaticArray( const StaticArray& v );

   /**
    * \brief Templated copy constructor.
    *
    * Constructs a copy of another static array \e v where the value type is
    * \e ValueType. This constructor allows casting between `StaticArray`
    * types with the same size, but different value types.
    */
   template< typename OtherValue >
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   constexpr StaticArray( const StaticArray< Size, OtherValue >& v );

   /**
    * \brief Move constructor.
    */
   constexpr StaticArray( StaticArray&& ) noexcept = default;

   /**
    * \brief Constructor which initializes the array element-by-element using
    * the supplied values.
    *
    * It behaves like the aggregate-initialization in \ref std::array, e.g.
    * `StaticArray< 3, int > a = {1, 2, 3};`. The number of supplied arguments
    * must be equal to \e Size.
    *
    * \param values The elements forwarded to the constructor of the underlying
    *               \ref std::array.
    */
   template< typename... Values, std::enable_if_t< ( Size > 1 ) && sizeof...( Values ) == Size, bool > = true >
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   constexpr StaticArray( Values&&... values );

   /**
    * \brief Constructor which initializes the array by copying elements from
    * \ref std::initializer_list, e.g. `{...}`.
    *
    * The initializer list size must larger or equal to \e Size.
    *
    * \param elems input initializer list
    */
   template< typename OtherValue >
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   constexpr StaticArray( const std::initializer_list< OtherValue >& elems );

   /**
    * \brief Constructor which initializes the array by copying elements from
    * \ref std::array.
    *
    * \param array input array
    */
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   template< typename OtherValue >
   __cuda_callable__
   constexpr StaticArray( const std::array< OtherValue, Size >& array );

   /**
    * \brief Constructor which initializes the array by moving elements from
    * \ref std::array.
    *
    * \param array input array
    */
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   constexpr StaticArray( std::array< Value, Size >&& array );

   /**
    * \brief Gets pointer to data of this static array.
    */
   [[nodiscard]] constexpr Value*
   getData() noexcept;

   /**
    * \brief Gets constant pointer to data of this static array.
    */
   [[nodiscard]] constexpr const Value*
   getData() const noexcept;

   /**
    * \brief Accesses specified element at the position \e i and returns a constant reference to its value.
    *
    * \param i Index position of an element.
    */
   [[nodiscard]] constexpr const Value&
   operator[]( int i ) const noexcept;

   /**
    * \brief Accesses specified element at the position \e i and returns a reference to its value.
    *
    * \param i Index position of an element.
    */
   [[nodiscard]] constexpr Value&
   operator[]( int i ) noexcept;

   /**
    * \brief Accesses specified element at the position \e i and returns a constant reference to its value.
    *
    * Equivalent to \ref operator[].
    */
   [[nodiscard]] constexpr const Value&
   operator()( int i ) const noexcept;

   /**
    * \brief Accesses specified element at the position \e i and returns a reference to its value.
    *
    * Equivalent to \ref operator[].
    */
   [[nodiscard]] constexpr Value&
   operator()( int i ) noexcept;

   /**
    * \brief Returns reference to the first coordinate.
    */
   [[nodiscard]] constexpr Value&
   x() noexcept;

   /**
    * \brief Returns constant reference to the first coordinate.
    */
   [[nodiscard]] constexpr const Value&
   x() const noexcept;

   /**
    * \brief Returns reference to the second coordinate for arrays with Size >= 2.
    */
   [[nodiscard]] constexpr Value&
   y() noexcept;

   /**
    * \brief Returns constant reference to the second coordinate for arrays with Size >= 2.
    */
   [[nodiscard]] constexpr const Value&
   y() const noexcept;

   /**
    * \brief Returns reference to the third coordinate for arrays with Size >= 3.
    */
   [[nodiscard]] constexpr Value&
   z() noexcept;

   /**
    * \brief Returns constant reference to the third coordinate for arrays with Size >= 3.
    */
   [[nodiscard]] constexpr const Value&
   z() const noexcept;

   /**
    * \brief Sets the value of the \e i-th element to \e value.
    *
    * \param i The index of the element to be set.
    * \param value The new value of the element.
    */
   constexpr void
   setElement( IndexType i, ValueType value ) noexcept;

   /**
    * \brief Returns the value of the \e i-th element.
    *
    * \param i The index of the element to be returned.
    */
   [[nodiscard]] constexpr Value
   getElement( IndexType i ) const noexcept;

   /**
    * \brief Copy-assignment operator.
    */
   // NOTE: as of nvcc 11.8, the default/implicit copy-assignment operator for std::array (which is based on aggregate
   // initialization) does not work correctly in device code, so we must define our own copy-constructor
   constexpr StaticArray&
   operator=( const StaticArray& v );

   /**
    * \brief Move-assignment operator.
    */
   constexpr StaticArray&
   operator=( StaticArray&& ) noexcept = default;

   /**
    * \brief Assigns an object \e v of type \e T.
    *
    * T can be:
    *
    * 1. Static linear container implementing operator[] and having the same size.
    * In this case, \e v is copied to this array elementwise.
    * 2. An object that can be converted to \e Value type. In this case all elements
    * are set to \e v.
    */
   template< typename T >
   constexpr StaticArray< Size, Value >&
   operator=( const T& v );

   /**
    * \brief This function checks whether this static array is equal to another \e array.
    *
    * Return \e true if the arrays are equal in size. Otherwise returns \e false.
    */
   template< typename Array >
   [[nodiscard]] constexpr bool
   operator==( const Array& array ) const;

   /**
    * \brief This function checks whether this static array is not equal to another \e array.
    *
    * Return \e true if the arrays are not equal in size. Otherwise returns \e false.
    */
   template< typename Array >
   [[nodiscard]] constexpr bool
   operator!=( const Array& array ) const;

   /**
    * \brief Sets all values of this static array to \e val.
    */
   constexpr void
   setValue( const ValueType& val );

   /**
    * \brief Saves this static array into the \e file.
    * \param file Reference to a file.
    */
   void
   save( File& file ) const;

   /**
    * \brief Loads data from the \e file to this static array.
    * \param file Reference to a file.
    */
   void
   load( File& file );

   /**
    * \brief Sorts the elements in this static array in ascending order.
    */
   constexpr void
   sort();

   /**
    * \brief Writes the array values into stream \e str with specified \e separator.
    *
    * @param str Reference to a stream.
    * @param separator Character separating the array values in the stream \e str.
    * Is set to " " by default.
    */
   std::ostream&
   write( std::ostream& str, const char* separator = " " ) const;

protected:
   std::array< Value, Size > data;
};

template< int Size, typename Value >
std::ostream&
operator<<( std::ostream& str, const StaticArray< Size, Value >& a );

/**
 * \brief Serialization of static arrays into binary files.
 *
 * \param file output file
 * \param array is an array to be written into the output file.
 */
template< int Size, typename Value >
File&
operator<<( File& file, const StaticArray< Size, Value >& array );

/**
 * \brief Serialization of static arrays into binary files.
 *
 * \param file output file
 * \param array is an array to be written into the output file.
 */
template< int Size, typename Value >
File&
operator<<( File&& file, const StaticArray< Size, Value >& array );

/**
 * \brief Deserialization of static arrays from binary files.
 *
 * \param file input file
 * \param array is an array to be read from the input file.
 */
template< int Size, typename Value >
File&
operator>>( File& file, StaticArray< Size, Value >& array );

/**
 * \brief Deserialization of static arrays from binary files.
 *
 * \param file input file
 * \param array is an array to be read from the input file.
 */
template< int Size, typename Value >
File&
operator>>( File&& file, StaticArray< Size, Value >& array );

}  // namespace TNL::Containers

// specializations to make StaticArray work with C++17 structured bindings
// (all these specializations exist for std::array)
namespace std {

template< int N, class T >
struct tuple_size< TNL::Containers::StaticArray< N, T > > : std::integral_constant< std::size_t, N >
{};

template< std::size_t I, int N, class T >
struct tuple_element< I, TNL::Containers::StaticArray< N, T > >
{
   using type = T;
};

}  // namespace std

// the `get` function must be defined in the TNL::Containers namespace,
// because structured binding finds it by ADL
namespace TNL::Containers {

template< std::size_t I, int N, class T >
constexpr T&
get( StaticArray< N, T >& a ) noexcept
{
   static_assert( I < N );
   return a[ I ];
}

template< std::size_t I, int N, class T >
constexpr T&&
get( StaticArray< N, T >&& a ) noexcept
{
   static_assert( I < N );
   return std::move( a[ I ] );
}

template< std::size_t I, int N, class T >
constexpr const T&
get( const StaticArray< N, T >& a ) noexcept
{
   static_assert( I < N );
   return a[ I ];
}

template< std::size_t I, int N, class T >
constexpr const T&&
get( const StaticArray< N, T >&& a ) noexcept
{
   static_assert( I < N );
   return std::move( a[ I ] );
}

}  // namespace TNL::Containers

#include <TNL/Containers/StaticArray.hpp>
