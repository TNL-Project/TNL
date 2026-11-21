// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>

namespace TNL::Algorithms::Sorting::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )

template< typename Value, typename Index, typename Device, typename Compare >
__device__
std::pair< Value, Index >
pickMedianOfThree( TNL::Containers::ArrayView< Value, Device, Index > src, const Compare& compare )
{
   if( src.getSize() <= 1 )
      return { src[ 0 ], 0 };

   const Value& a = src[ 0 ];
   const Value& b = src[ src.getSize() / 2 ];
   const Value& c = src[ src.getSize() - 1 ];

   if( compare( a, b ) )  // ..a..b..
   {
      if( compare( b, c ) )  // ..a..b..c
         return { b, src.getSize() / 2 };
      else if( compare( c, a ) )  //..c..a..b..
         return { a, 0 };
      else  //..a..c..b..
         return { c, src.getSize() - 1 };
   }
   else  //..b..a..
   {
      if( compare( a, c ) )  //..b..a..c
         return { a, 0 };
      else if( compare( c, b ) )  //..c..b..a..
         return { b, src.getSize() / 2 };
      else  //..b..c..a..
         return { c, src.getSize() - 1 };
   }
}

#endif

}  // namespace TNL::Algorithms::Sorting::detail
