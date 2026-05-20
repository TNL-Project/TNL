// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Math.h>
#include <limits>
#include <type_traits>

namespace TNL::Algorithms::Sorting::detail {

template< typename T >
__cuda_callable__
T
closestPow2( T x )
{
   static_assert( std::is_integral_v< T > );

   if( x == 0 )
      return 0;

   T ret = 1;
   while( ret < x )
      ret <<= 1;

   return ret;
}

#if defined( __CUDACC__ )
// Inline PTX call to return index of highest non-zero bit in a word
__device__
__forceinline__ unsigned int
__btflo( unsigned int word )
{
   unsigned int ret;
   asm volatile( "bfind.u32 %0, %1;" : "=r"( ret ) : "r"( word ) );
   return ret;
}

template< typename T >
__device__
std::common_type_t< T, unsigned int >
closestPow2_ptx( T bitonicLen )
{
   static_assert( std::is_integral_v< T > );

   if constexpr( sizeof( T ) <= sizeof( unsigned int ) ) {
      return 1U << ( __btflo( static_cast< unsigned int >( bitonicLen ) - 1U ) + 1 );
   }
   else {
      if( bitonicLen <= static_cast< T >( std::numeric_limits< unsigned int >::max() ) ) {
         return closestPow2_ptx( static_cast< unsigned int >( bitonicLen ) );
      }
      else {
         // For values larger than unsigned int, use standard implementation
         return closestPow2( bitonicLen );
      }
   }
}
#elif defined( __HIP__ )
// TODO: optimize this for AMD GPUs
template< typename T >
__device__
T
closestPow2_ptx( T bitonicLen )
{
   static_assert( std::is_integral_v< T > );
   return closestPow2( bitonicLen );
}
#endif

}  // namespace TNL::Algorithms::Sorting::detail
