// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Sorting/detail/bitonicSort.h>

namespace TNL::Algorithms::Sorting {

struct BitonicSort
{
   template< typename Array >
   void static sort( Array& array )
   {
      sort( array, std::less<>{} );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      detail::bitonicSort( array.getView(), compare );
   }

   template< typename Device, typename Index, typename Compare, typename Swap >
   void static inplaceSort( const Index begin, const Index end, const Compare& compare, const Swap& swap )
   {
      static_assert( std::is_same_v< Device, Devices::GPU >, "inplace bitonic sort is implemented only for GPU" );
      detail::bitonicSort( begin, end, compare, swap );
   }
};

}  // namespace TNL::Algorithms::Sorting
