// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "detail/Quicksort.h"

namespace TNL::Algorithms::Sorting {

/**
 * \brief Parallel quicksort.
 *
 * The sort is not stable and needs O( N ) auxiliary storage. The implementation
 * is available for the GPU devices only; instantiating the sort for another
 * device type reports a compilation error instead of falling back silently.
 *
 * \tparam Array type of the array to be sorted. Its value type, device type and
 *               index type determine the instantiation of the implementation.
 */
struct Quicksort
{
   template< typename Array >
   void static sort( Array& array )
   {
      detail::Quicksort< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType > qs;
      qs.sort( array );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      detail::Quicksort< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType > qs;
      qs.sort( array, compare );
   }
};

}  // namespace TNL::Algorithms::Sorting
