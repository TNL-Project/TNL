// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Containers/ndarray/Meta.h>

namespace TNL::Containers {

/**
 * \brief Holds static sizes of an N-dimensional array.
 *
 * The difference from \ref SizesHolder is that zero value in \e sizes
 * does not indicate a dynamic value, the corresponding size is always
 * zero (both static and dynamic).
 *
 * \tparam Index Integral type used for the representation of sizes.
 * \tparam sizes Sequence of integers specifying static sizes. The number of
 *         integers in the sequence specifies the dimension of the array.
 *
 * \ingroup ndarray
 */
template< typename Index, Index... sizes >
class StaticSizesHolder
{
public:
   using IndexType = Index;

   //! \brief Default constructor.
   StaticSizesHolder() = default;

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
      static_assert( level < getDimension(), "Invalid dimension passed to getStaticSize()." );
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
   //! It is always equal to the _static_ size.
   template< std::size_t level >
   [[nodiscard]] __cuda_callable__
   Index
   getSize() const
   {
      static_assert( level < getDimension(), "Invalid dimension passed to getSize()." );
      return getStaticSize< level >();
   }

   //! \brief Returns the _dynamic_ size along a specific axis.
   //! It is always equal to the _static_ size.
   [[nodiscard]] __cuda_callable__
   Index
   operator[]( Index level ) const
   {
      TNL_ASSERT_GE( level, 0, "Invalid dimension passed to operator[]." );
      TNL_ASSERT_LT( level, static_cast< Index >( sizeof...( sizes ) ), "Invalid dimension passed to operator[]." );
      return getStaticSize( level );
   }
};

// helper for the methods forAll, forInterior, etc.
template< typename Index, std::size_t dimension, Index constSize >
class ConstStaticSizesHolder
{
public:
   using IndexType = Index;

   [[nodiscard]] static constexpr std::size_t
   getDimension()
   {
      return dimension;
   }

   template< std::size_t level >
   [[nodiscard]] static constexpr std::size_t
   getStaticSize()
   {
      static_assert( level < getDimension(), "Invalid dimension passed to getStaticSize()." );
      return constSize;
   }

   //! \brief Returns the _static_ size of a specific dimension identified by
   //! a _runtime_ parameter \e level.
   [[nodiscard]] static constexpr Index
   getStaticSize( Index level )
   {
      return constSize;
   }

   template< std::size_t level >
   [[nodiscard]] __cuda_callable__
   Index
   getSize() const
   {
      static_assert( level < getDimension(), "Invalid dimension passed to getSize()." );
      return constSize;
   }

   //! \brief Returns the _dynamic_ size along a specific axis.
   //! It is always equal to the _static_ size.
   [[nodiscard]] __cuda_callable__
   Index
   operator[]( Index level ) const
   {
      TNL_ASSERT_GE( level, 0, "Invalid dimension passed to operator[]." );
      TNL_ASSERT_LT( level, static_cast< Index >( getDimension() ), "Invalid dimension passed to operator[]." );
      return constSize;
   }
};

}  // namespace TNL::Containers
