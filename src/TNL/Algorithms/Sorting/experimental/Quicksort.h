// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "detail/Quicksorter.h"

namespace TNL::Algorithms::Sorting {

/**
 * \ingroup experimental
 * \experimental
 * \brief  Namespace for experimental sorting algorithms.
 */
namespace experimental {

/**
 * \ingroup experimental
 * \experimental
 * \brief Parallel quicksort for CUDA.
 *
 */
struct Quicksort
{
   template< typename Array >
   void static sort( Array& array )
   {
      detail::Quicksorter< typename Array::ValueType, typename Array::DeviceType > qs;
      qs.sort( array );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      detail::Quicksorter< typename Array::ValueType, typename Array::DeviceType > qs;
      qs.sort( array, compare );
   }
};

}  // namespace experimental
}  // namespace TNL::Algorithms::Sorting
