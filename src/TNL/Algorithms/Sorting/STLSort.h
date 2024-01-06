// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>

namespace TNL::Algorithms::Sorting {

struct STLSort
{
   template< typename Array >
   void static sort( Array& array )
   {
      std::sort( array.getData(), array.getData() + array.getSize() );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      std::sort( array.getData(), array.getData() + array.getSize(), compare );
   }
};

}  // namespace TNL::Algorithms::Sorting
