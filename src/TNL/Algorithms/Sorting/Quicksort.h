// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Sorting/detail/Quicksorter.h>

namespace TNL::Algorithms::Sorting {

struct Quicksort
{
   template< typename Array >
   void static sort( Array& array )
   {
      Quicksorter< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType > qs;
      qs.sort( array );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      Quicksorter< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType > qs;
      qs.sort( array, compare );
   }
};

}  // namespace TNL::Algorithms::Sorting
