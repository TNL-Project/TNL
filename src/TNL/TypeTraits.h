// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <type_traits>
#include <utility>

namespace TNL {

template< class T >
struct is_complex : public std::false_type
{};
template< class T >
struct is_complex< const T > : public is_complex< T >
{};
template< class T >
struct is_complex< volatile const T > : public is_complex< T >
{};
template< class T >
struct is_complex< volatile T > : public is_complex< T >
{};
template< class T >
struct is_complex< std::complex< T > > : public std::true_type
{};

template< class T >
constexpr bool is_complex_v = is_complex< T >::value;

// clang-format off

template< typename T, typename R = void >
struct enable_if_type
{
   using type = R;
};

/**
 * \brief Type trait for checking if T has getArrayData method.
 */
template< typename T >
class HasGetArrayDataMethod
{
private:
   using YesType = char[1];
   using NoType = char[2];

   template< typename C > static YesType& test( decltype(std::declval< C >().getArrayData()) );
   template< typename C > static NoType& test(...);

public:
   static constexpr bool value = ( sizeof( test< std::decay_t< T > >( nullptr ) ) == sizeof( YesType ) );
};

/**
 * \brief Type trait for checking if T has getSize method.
 */
template< typename T >
class HasGetSizeMethod
{
private:
   using YesType = char[1];
   using NoType = char[2];

   template< typename C > static YesType& test( decltype(std::declval< C >().getSize() ) );
   template< typename C > static NoType& test(...);

public:
   static constexpr bool value = ( sizeof( test< std::decay_t<T> >(0) ) == sizeof( YesType ) );
};

/**
 * \brief Type trait for checking if T has setSize method.
 */
template< typename T >
class HasSetSizeMethod
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> std::enable_if_t<
         std::is_same_v< decltype( std::declval<U>().setSize(0) ), void >,
         std::true_type
      >;

   template< typename >
   static constexpr std::false_type check(...);

   using type = decltype( check< std::decay_t< T > >( nullptr ) );

public:
   static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T has operator[] taking one index argument.
 */
template< typename T >
class HasSubscriptOperator
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> std::enable_if_t<
         // NOLINTNEXTLINE(readability-static-accessed-through-instance)
         ! std::is_same_v< decltype( std::declval<U>()[ std::declval<U>().getSize() ] ), void >,
         std::true_type
      >;

   template< typename >
   static constexpr std::false_type check(...);

   using type = decltype( check< std::decay_t< T > >( nullptr ) );

public:
   static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T has operator+= taking one argument of type T.
 */
template< typename T >
class HasAddAssignmentOperator
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> std::enable_if_t<
         ! std::is_same_v< decltype( std::declval<U>() += std::declval<U>() ), void >,
         std::true_type
      >;

   template< typename >
   static constexpr std::false_type check(...);

   using type = decltype( check< std::decay_t< T > >( nullptr ) );

public:
   static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T is a [scalar type](https://en.wikipedia.org/wiki/Scalar_(mathematics))
 * (in the mathemtatical sense). Not to be confused with \ref std::is_scalar.
 *
 * For example, \ref std::is_arithmetic "arithmetic types" as defined by the STL
 * are scalar types. \ref std::complex is also considered as scalar type.
 * TNL also provides additional scalar types, e.g. for extended precision
 * arithmetics. Users may also define specializations of this trait class for
 * their custom scalar types.
 */
template< typename T >
struct IsScalarType
: public std::integral_constant< bool, std::is_arithmetic_v< T > || is_complex_v< T > >
{};

/**
 * \brief Type trait for checking if T is an array type, e.g.
 *        \ref Containers::Array or \ref Containers::Vector.
 *
 * The trait combines \ref HasGetArrayDataMethod, \ref HasGetSizeMethod,
 * and \ref HasSubscriptOperator.
 */
template< typename T >
struct IsArrayType
: public std::integral_constant< bool,
            HasGetArrayDataMethod< T >::value &&
            HasGetSizeMethod< T >::value &&
            HasSubscriptOperator< T >::value >
{};

/**
 * \brief Type trait for checking if T is a vector type, e.g.
 *        \ref Containers::Vector or \ref Containers::VectorView.
 */
template< typename T >
struct IsVectorType
: public std::integral_constant< bool,
            IsArrayType< T >::value &&
            HasAddAssignmentOperator< T >::value >
{};

/**
 * \brief Type trait for checking if T has a \e constexpr \e getSize method.
 */
template< typename T >
struct HasConstexprGetSizeMethod
{
private:
   // implementation adopted from here: https://stackoverflow.com/a/50169108
   template< bool hasGetSize = HasGetSizeMethod< T >::value, typename = void >
   struct impl
   {
      // disable nvcc warning: invalid narrowing conversion from "unsigned int" to "int"
      // (the implementation is based on the conversion)
      #ifdef __NVCC__
         #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
            #pragma nv_diagnostic push
            #pragma nv_diag_suppress 2361
         #else
            #pragma push
            #pragma diag_suppress 2361
         #endif
      #endif
      template< typename M, M method >
      static constexpr std::true_type is_constexpr_impl( decltype(int{((*method)(), 0U)}) );
      #ifdef __NVCC__
         #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
            #pragma nv_diagnostic pop
         #else
            #pragma pop
         #endif
      #endif

      template< typename M, M method >
      static constexpr std::false_type is_constexpr_impl(...);

      using type = decltype(is_constexpr_impl< decltype(&std::decay_t<T>::getSize), &std::decay_t<T>::getSize >(0));
   };

   // specialization for types which don't have getSize() method at all
   template< typename _ >
   struct impl< false, _ >
   {
      using type = std::false_type;
   };

   using type = typename impl<>::type;

public:
   static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T is a static array type.
 *
 * Static array types are array types which have a \e constexpr \e getSize
 * method.
 */
template< typename T >
struct IsStaticArrayType
: public std::integral_constant< bool,
            HasConstexprGetSizeMethod< T >::value &&
            HasSubscriptOperator< T >::value >
{};

/**
 * \brief Type trait for checking if T is a view type.
 */
template< typename T >
struct IsViewType
{
private:
   template< typename C >
   static constexpr auto test(C) -> std::integral_constant< bool, std::is_same_v< typename C::ViewType, C > >;

   static constexpr std::false_type test(...);

public:
   static constexpr bool value = decltype( test(std::decay_t<T>{}) )::value;
};

/**
 * \brief Type trait for checking if T has getCommunicator method.
 */
template< typename T >
class HasGetCommunicatorMethod
{
private:
   using YesType = char[1];
   using NoType = char[2];

   template< typename C > static YesType& test( decltype(std::declval< C >().getCommunicator()) );
   template< typename C > static NoType& test(...);

public:
   // NOLINTNEXTLINE(modernize-use-nullptr)
   static constexpr bool value = ( sizeof( test< std::decay_t<T> >(0) ) == sizeof( YesType ) );
};

/**
 * \brief Copy const qualifier from Source type to Target type.
 */
template< typename Target >
struct copy_const
{
   template< typename Source >
   struct from
   {
      using type = std::conditional_t<
         std::is_const_v< Source >,
         std::add_const_t< Target >, Target >;
   };
};

/**
 * \brief Type trait for checking if T has count member
 */
template< typename T >
class HasCountMember
{
private:
   using YesType = char[1];
   using NoType = char[2];

   template< typename C > static YesType& test( decltype( &C::count ) );
   template< typename C > static NoType& test(...);

public:
   static constexpr bool value = ( sizeof( test< std::decay_t<T> >(0) ) == sizeof( YesType ) );  // NOLINT(readability-implicit-bool-conversion, modernize-use-nullptr)
};

// clang-format on

/**
 * \brief Get the underlying value type of `T`.
 *
 * This recursively descends into the `::ValueType` or `::value_type` local
 * type aliases and returns the underlying value type. For example, if a vector
 * type such as
 *
 * ```cpp
 * TNL::Containers::StaticVector< 1, TNL::Arithmetics::Complex< double > >
 * ```
 *
 * is given as `T`, this will return `double`.
 */
template< typename T >
struct GetValueType
{
private:
   template< typename TT, bool IsArithmetic = std::is_arithmetic_v< T > >
   struct impl;

   template< typename TT >
   struct impl< std::complex< TT >, false >
   {
      using type = TT;
   };

   template< typename TT >
   struct impl< TT, true >
   {
      using type = TT;
   };

   template< typename TT >
   struct impl< TT, false >
   {
      using type = typename GetValueType< typename T::ValueType >::type;
   };

public:
   using type = typename impl< T >::type;
};

template< typename T >
using GetValueType_t = typename GetValueType< T >::type;

}  // namespace TNL
