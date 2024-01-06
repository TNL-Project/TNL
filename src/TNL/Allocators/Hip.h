// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Exceptions/BackendBadAlloc.h>
#include <TNL/Exceptions/BackendSupportMissing.h>
#include <TNL/Backend/Macros.h>
#include "Traits.h"

namespace TNL::Allocators {

/**
 * \brief Allocator for the HIP device memory space.
 *
 * The allocation is done using the `hipMalloc` function and the deallocation
 * is done using the `hipFree` function.
 */
template< class T >
struct Hip
{
   using value_type = T;
   using size_type = std::size_t;
   using difference_type = std::ptrdiff_t;

   Hip() = default;
   Hip( const Hip& ) = default;
   Hip( Hip&& ) noexcept = default;

   Hip&
   operator=( const Hip& ) = default;
   Hip&
   operator=( Hip&& ) noexcept = default;

   template< class U >
   Hip( const Hip< U >& )
   {}

   template< class U >
   Hip( Hip< U >&& )
   {}

   template< class U >
   Hip&
   operator=( const Hip< U >& )
   {
      return *this;
   }

   template< class U >
   Hip&
   operator=( Hip< U >&& )
   {
      return *this;
   }

   [[nodiscard]] value_type*
   allocate( size_type n )
   {
#ifdef __HIP__
      value_type* result = nullptr;
      if( hipMalloc( (void**) &result, n * sizeof( value_type ) ) != hipSuccess )
         throw Exceptions::BackendBadAlloc();
      return result;
#else
      throw Exceptions::BackendSupportMissing();
#endif
   }

   void
   deallocate( value_type* ptr, size_type )
   {
#ifdef __HIP__
      TNL_BACKEND_SAFE_CALL( hipFree( (void*) ptr ) );
#else
      throw Exceptions::BackendSupportMissing();
#endif
   }
};

template< class T1, class T2 >
[[nodiscard]] bool
operator==( const Hip< T1 >&, const Hip< T2 >& )
{
   return true;
}

template< class T1, class T2 >
[[nodiscard]] bool
operator!=( const Hip< T1 >& lhs, const Hip< T2 >& rhs )
{
   return ! ( lhs == rhs );
}

}  // namespace TNL::Allocators

namespace TNL {
template< class T >
struct allocates_host_accessible_data< Allocators::Hip< T > > : public std::false_type
{};
}  // namespace TNL
