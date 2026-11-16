// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Math.h>
#include <type_traits>

namespace TNL::Algorithms::Sorting::detail {

/**
 * \brief Returns the base-2 logarithm of the closest power of two that is not
 * smaller than `x`.
 *
 * The result is \f$ \lceil \log_2 x \rceil \f$ for \f$ x > 1 \f$ and zero
 * otherwise. Bitonic sort phases are always powers of two, so the logarithms
 * are used to replace integer division by bit shifts in the kernels.
 */
template< typename T >
__cuda_callable__
unsigned int
closestPow2Log2( T x )
{
   static_assert( std::is_integral_v< T > );

   if( x <= 1 )
      return 0;

   unsigned int log = 0;
   T y = static_cast< T >( x - 1 );
   while( y > 0 ) {
      y >>= 1;
      ++log;
   }
   return log;
}

}  // namespace TNL::Algorithms::Sorting::detail
