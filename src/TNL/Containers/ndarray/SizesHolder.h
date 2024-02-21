// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Algorithms/staticFor.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/ndarray/Meta.h>

namespace TNL::Containers {

/**
 * \brief Holds static and dynamic sizes of an N-dimensional array.
 *
 * The dimension of the array and static sizes are specified as
 * \ref std::size_t, the type of dynamic sizes is configurable with \e Index.
 *
 * \tparam Index Integral type used for storing dynamic sizes.
 * \tparam sizes Sequence of integers specifying static and dynamic sizes. The
 *         number of integers in the sequence specifies the dimension of the
 *         array.  Positive values specify static sizes, zeros specify dynamic
 *         sizes that must be set at run-time via \ref setSize.
 *
 * \ingroup ndarray
 */
template< typename Index, std::size_t... sizes >
class SizesHolder
{
public:
   using IndexType = Index;

   //! \brief Default constructor.
   constexpr SizesHolder()
   {
      dynamicSizes.setValue( 0 );
   }

   //! \brief Constructs the holder from given pack of sizes.
   template< typename... Indices, std::enable_if_t< sizeof...( Indices ) == sizeof...( sizes ), bool > = true >
   explicit SizesHolder( Indices... _sizes )
   {
      Algorithms::staticFor< std::size_t, 0, getDimension() >(
         [ & ]( auto i )
         {
            setSize< i >( detail::get_from_pack< i >( _sizes... ) );
         } );
   }

   //! \brief Returns the dimension of the array, i.e. number of \e sizes
   //! specified in the template parameters.
   [[nodiscard]] static constexpr std::size_t
   getDimension()
   {
      return sizeof...( sizes );
   }

   //! \brief Returns the _static_ size of a specific dimension.
   template< std::size_t level >
   [[nodiscard]] static constexpr std::size_t
   getStaticSize()
   {
      static_assert( level < sizeof...( sizes ), "Invalid dimension passed to getStaticSize()." );
      return detail::get_from_pack< level >( sizes... );
   }

   //! \brief Returns the _static_ size of a specific dimension identified by
   //! a _runtime_ parameter \e level.
   [[nodiscard]] static constexpr Index
   getStaticSize( Index level )
   {
      Index result = 0;
      Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) >(
         [ &result, level ]( auto i )
         {
            if( i == level )
               result = getStaticSize< i >();
         } );
      return result;
   }

   //! \brief Returns the _dynamic_ size along a specific axis.
   template< std::size_t level >
   [[nodiscard]] __cuda_callable__
   Index
   getSize() const
   {
      static_assert( level < sizeof...( sizes ), "Invalid dimension passed to getSize()." );
      if constexpr( getStaticSize< level >() > 0 ) {
         return getStaticSize< level >();
      }
      else {
         constexpr std::size_t idx = getDynamicSizeIndex( level );
         return dynamicSizes[ idx ];
      }
   }

   //! \brief Sets the _dynamic_ size along a specific axis.
   template< std::size_t level >
   __cuda_callable__
   void
   setSize( Index size )
   {
      static_assert( level < sizeof...( sizes ), "Invalid dimension passed to setSize()." );
      if constexpr( getStaticSize< level >() > 0 ) {
         TNL_ASSERT_EQ( size, 0, "Dynamic size for a static dimension must be 0." );
      }
      else {
         constexpr std::size_t idx = getDynamicSizeIndex( level );
         dynamicSizes[ idx ] = size;
      }
   }

   /**
    * \brief Dynamic accessor for the _dynamic_ size along a specific axis.
    *
    * **Warning:** The static size of given level must be equal to zero.
    *
    * **Note:** The access is less efficient compared to the \ref getSize and
    * \ref setSize methods, since the mapping from \e level to the dynamic
    * storage must be computed at runtime rather than compile-time.
    */
   [[nodiscard]] __cuda_callable__
   const Index&
   operator[]( Index level ) const
   {
      TNL_ASSERT_GE( level, 0, "Invalid dimension passed to getSize()." );
      TNL_ASSERT_LT( level, static_cast< Index >( sizeof...( sizes ) ), "Invalid dimension passed to getSize()." );
      TNL_ASSERT_EQ( getStaticSize( level ), 0, "The static size of given dimension must be equal to zero." );

      const std::size_t idx = getDynamicSizeIndex( level );
      return dynamicSizes[ idx ];
   }

   /**
    * \brief Dynamic non-const accessor for the _dynamic_ size along a specific axis.
    *
    * **Warning:** The static size of given level must be equal to zero.
    *
    * **Note:** The access is less efficient compared to the \ref getSize and
    * \ref setSize methods, since the mapping from \e level to the dynamic
    * storage must be computed at runtime rather than compile-time.
    */
   [[nodiscard]] __cuda_callable__
   Index&
   operator[]( Index level )
   {
      TNL_ASSERT_GE( level, 0, "Invalid dimension passed to operator[]." );
      TNL_ASSERT_LT( level, static_cast< Index >( sizeof...( sizes ) ), "Invalid dimension passed to operator[]." );
      TNL_ASSERT_EQ( getStaticSize( level ), 0, "The static size of given dimension must be equal to zero." );

      const std::size_t idx = getDynamicSizeIndex( level );
      return dynamicSizes[ idx ];
   }

   //! \brief Compares the sizes with another instance of the holder.
   [[nodiscard]] __cuda_callable__
   bool
   operator==( const SizesHolder& other ) const
   {
      return dynamicSizes == other.dynamicSizes;
   }

   //! \brief Compares the sizes with another instance of the holder.
   [[nodiscard]] __cuda_callable__
   bool
   operator!=( const SizesHolder& other ) const
   {
      return ! operator==( other );
   }

protected:
   //! \brief Checks if given \e level corresponds to a static size.
   template< std::size_t level >
   [[nodiscard]] static constexpr bool
   isStaticSize()
   {
      return getStaticSize< level >() > 0;
   }

   //! \brief Returns the number of dynamic sizes that need to be stored at runtime.
   [[nodiscard]] static constexpr std::size_t
   countDynamicSizes()
   {
      std::size_t count = 0;
      Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) >(
         [ &count ]( auto i )
         {
            if( ! isStaticSize< i >() )
               count++;
         } );
      return count;
   }

   /**
    * \brief Returns the index of given \e level in the array of dynamic sizes.
    *
    * **WARNING:** \e level must correspond to a dynamic size, otherwise the
    * \e level should **not** be used for indexing the array of dynamic sizes.
    * This must be ensured before calling this method - it can't be checked by
    * a `static_assert` here, because we want to be able to specify the
    * \e level it at runtime as well.
    */
   [[nodiscard]] static constexpr std::size_t
   getDynamicSizeIndex( std::size_t level )
   {
      std::size_t result = 0;
      Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) >(
         [ &result, level ]( auto i )
         {
            if( i < level && ! isStaticSize< i >() )
               result++;
         } );
      return result;
   }

   StaticArray< countDynamicSizes(), Index > dynamicSizes;
};

/**
 * \brief Combines the sizes of two instance of \ref SizesHolder with the operator `+`.
 *
 * \ingroup ndarray
 */
template< typename Index, std::size_t... sizes, typename OtherHolder >
SizesHolder< Index, sizes... >
operator+( const SizesHolder< Index, sizes... >& lhs, const OtherHolder& rhs )
{
   SizesHolder< Index, sizes... > result;
   Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) >(
      [ &result, &lhs, &rhs ]( auto level )
      {
         if( result.template getStaticSize< level >() == 0 )
            result.template setSize< level >( lhs.template getSize< level >() + rhs.template getSize< level >() );
      } );
   return result;
}

/**
 * \brief Combines the sizes of two instance of \ref SizesHolder with the operator `-`.
 *
 * \ingroup ndarray
 */
template< typename Index, std::size_t... sizes, typename OtherHolder >
SizesHolder< Index, sizes... >
operator-( const SizesHolder< Index, sizes... >& lhs, const OtherHolder& rhs )
{
   SizesHolder< Index, sizes... > result;
   Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) >(
      [ &result, &lhs, &rhs ]( auto level )
      {
         if( result.template getStaticSize< level >() == 0 )
            result.template setSize< level >( lhs.template getSize< level >() - rhs.template getSize< level >() );
      } );
   return result;
}

/**
 * \brief Prints the sizes contained in an instance of \ref SizesHolder to the
 * given output stream.
 *
 * \ingroup ndarray
 */
template< typename Index, std::size_t... sizes >
std::ostream&
operator<<( std::ostream& str, const SizesHolder< Index, sizes... >& holder )
{
   str << "SizesHolder< ";
   Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) - 1 >(
      [ &str, &holder ]( auto dimension )
      {
         str << holder.template getStaticSize< dimension >() << ", ";
      } );
   str << holder.template getStaticSize< sizeof...( sizes ) - 1 >() << " >( ";
   Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) - 1 >(
      [ &str, &holder ]( auto dimension )
      {
         str << holder.template getSize< dimension >() << ", ";
      } );
   str << holder.template getSize< sizeof...( sizes ) - 1 >() << " )";
   return str;
}

namespace detail {

// helper for the forInterior method
template< typename SizesHolder, std::size_t ConstValue >
struct SubtractedSizesHolder
{};

template< typename Index, std::size_t ConstValue, std::size_t... sizes >
struct SubtractedSizesHolder< SizesHolder< Index, sizes... >, ConstValue >
{
   // using type = SizesHolder< Index, std::max( (std::size_t) 0, sizes - ConstValue )... >;
   using type = SizesHolder< Index, ( ( sizes >= ConstValue ) ? sizes - ConstValue : 0 )... >;
};

}  // namespace detail

}  // namespace TNL::Containers
