// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>

#include "detail/HostQuicksort.h"

namespace TNL::Algorithms::Sorting {

/**
 * \brief Parallel quicksort for the host devices.
 *
 * Three-way quicksort that sorts the ranges in place and uses OpenMP tasks to
 * sort the partitioned parts in parallel. It requires O( log N ) additional
 * memory for the recursion and no auxiliary array of the input size.
 *
 * \tparam Array type of the array to be sorted. Its value type, device type and
 *               index type determine the instantiation of the implementation.
 *               Only \ref TNL::Devices::Host and \ref TNL::Devices::Sequential
 *               are supported.
 */
struct HostQuicksort
{
   template< typename Array, typename Compare >
   static void
   sort( Array& array, const Compare& compare )
   {
      detail::HostQuicksort< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType >::sort(
         array, compare );
   }

   template< typename Array >
   static void
   sort( Array& array )
   {
      HostQuicksort::sort( array, std::less<>{} );
   }
};

}  // namespace TNL::Algorithms::Sorting
