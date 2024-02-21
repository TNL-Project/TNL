// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SizesHolder.h"

namespace TNL::Containers {

// wrapper for localBegins in DistributedNDArray (static sizes cannot be distributed, begins are always 0)
template< typename SizesHolder,
          // overridable value is useful in the forInterior method
          std::size_t ConstValue = 0 >
struct LocalBeginsHolder : public SizesHolder
{
   template< std::size_t dimension >
   [[nodiscard]] static constexpr std::size_t
   getStaticSize()
   {
      static_assert( dimension < SizesHolder::getDimension(), "Invalid dimension passed to getStaticSize()." );
      return ConstValue;
   }

   template< std::size_t level >
   [[nodiscard]] __cuda_callable__
   typename SizesHolder::IndexType
   getSize() const
   {
      if constexpr( SizesHolder::template getStaticSize< level >() != 0 )
         return ConstValue;
      else
         return SizesHolder::template getSize< level >();
   }

   template< std::size_t level >
   __cuda_callable__
   void
   setSize( typename SizesHolder::IndexType newSize )
   {
      if constexpr( SizesHolder::template getStaticSize< level >() == 0 )
         SizesHolder::template setSize< level >( newSize );
      else
         TNL_ASSERT_EQ( newSize,
                        (typename SizesHolder::IndexType) ConstValue,
                        "Dynamic size for a static dimension must be equal to the specified ConstValue." );
   }
};

template< typename SizesHolder, std::size_t ConstValue, typename OtherHolder >
LocalBeginsHolder< SizesHolder, ConstValue >
operator+( const LocalBeginsHolder< SizesHolder, ConstValue >& lhs, const OtherHolder& rhs )
{
   LocalBeginsHolder< SizesHolder, ConstValue > result;
   Algorithms::staticFor< std::size_t, 0, SizesHolder::getDimension() >(
      [ &result, &lhs, &rhs ]( auto level )
      {
         if constexpr( SizesHolder::template getStaticSize< level >() == 0 )
            result.template setSize< level >( lhs.template getSize< level >() + rhs.template getSize< level >() );
      } );
   return result;
}

template< typename SizesHolder, std::size_t ConstValue, typename OtherHolder >
LocalBeginsHolder< SizesHolder, ConstValue >
operator-( const LocalBeginsHolder< SizesHolder, ConstValue >& lhs, const OtherHolder& rhs )
{
   LocalBeginsHolder< SizesHolder, ConstValue > result;
   Algorithms::staticFor< std::size_t, 0, SizesHolder::getDimension() >(
      [ &result, &lhs, &rhs ]( auto level )
      {
         if constexpr( SizesHolder::template getStaticSize< level >() == 0 )
            result.template setSize< level >( lhs.template getSize< level >() - rhs.template getSize< level >() );
      } );
   return result;
}

template< typename SizesHolder, std::size_t ConstValue >
std::ostream&
operator<<( std::ostream& str, const LocalBeginsHolder< SizesHolder, ConstValue >& holder )
{
   str << "LocalBeginsHolder< SizesHolder< ";
   Algorithms::staticFor< std::size_t, 0, SizesHolder::getDimension() - 1 >(
      [ &str, &holder ]( auto dimension )
      {
         str << holder.template getStaticSize< dimension >() << ", ";
      } );
   str << holder.template getStaticSize< SizesHolder::getDimension() - 1 >() << " >, ";
   str << ConstValue << " >( ";
   Algorithms::staticFor< std::size_t, 0, SizesHolder::getDimension() - 1 >(
      [ &str, &holder ]( auto dimension )
      {
         str << holder.template getSize< dimension >() << ", ";
      } );
   str << holder.template getSize< SizesHolder::getDimension() - 1 >() << " )";
   return str;
}

}  // namespace TNL::Containers
