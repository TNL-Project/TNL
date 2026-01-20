// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/reduce.h>

namespace TNL::Algorithms {

// TODO: Add example to the documentation.
/**
 * \brief Find the first occurrence of a value in an array.
 *
 * \tparam Container is the type of the container.
 * \tparam ValueType is the type of the value to be found.
 * \tparam IndexType is the type used for indexing.
 * \param container is the array where the value is searched.
 * \param value is the value to be found.
 * \return pair `(found, position)` where \e found is a boolean indicating
 *         if the \e value was found and \e position is the position of the
 *         first occurrence in the container.
 */
template< typename Container, typename ValueType >
std::pair< bool, typename Container::IndexType >
find( const Container& container, const ValueType& value )
{
   static_assert( HasGetSizeMethod< Container >::value && HasSubscriptOperator< Container >::value );
   using IndexType = typename Container::IndexType;
   auto view = container.getConstView();
   auto fetch = [ view, value ] __cuda_callable__( IndexType i ) -> bool
   {
      return view[ i ] == value;
   };
   auto reduce = [] __cuda_callable__( bool& a, bool b, IndexType& aIdx, IndexType bIdx )
   {
      if( b ) {
         if( ! a ) {
            aIdx = bIdx;
            a = true;
         }
         else if( bIdx < aIdx ) {
            // ensure that the first occurrence is found
            aIdx = bIdx;
            // a is already true in this branch
         }
      }
   };
   return Algorithms::reduceWithArgument< typename Container::DeviceType >(
      (IndexType) 0, view.getSize(), fetch, reduce, false );
}

/**
 * \brief Find the first occurrence of a larger element than given value in sorted array.
 *
 * \param array is the array where the value is searched.
 * \param size is the size of the array.
 * \param value is the upper bound value.
 * \return pair `(found, position)` where \e found is a boolean indicating
 *         if the  upper bound \e value was found and \e position is the
 *         position in the array.
 */
template< typename Value, typename Index >
constexpr std::pair< bool, Index >
findUpperBound( const Value* array, Index size, const Value& value )
{
   if( array[ size - 1 ] <= value )
      return { false, size };
   Index left( 0 );
   Index right( size - 1 );
   while( left < right ) {
      Index mid = left + ( right - left ) / 2;
      if( array[ mid ] <= value )
         left = mid + 1;
      else
         right = mid;
   }
   return { true, left };
}

}  // namespace TNL::Algorithms
