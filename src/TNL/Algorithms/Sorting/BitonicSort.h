// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Sorting/detail/bitonicSort.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL::Algorithms::Sorting {

struct BitonicSort
{
   template< typename Array >
   void static sort( Array& array )
   {
      bitonicSort( array );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      bitonicSort( array, compare );
   }

   template< typename Device, typename Index, typename Compare, typename Swap >
   void static inplaceSort( const Index begin, const Index end, const Compare& compare, const Swap& swap )
   {
      if constexpr( std::is_same_v< Device, Devices::Cuda > )
         bitonicSort( begin, end, compare, swap );
      else
         throw Exceptions::NotImplementedError( "inplace bitonic sort is implemented only for CUDA" );
   }
};

}  // namespace TNL::Algorithms::Sorting
