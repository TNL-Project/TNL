// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Exceptions/BackendBadAlloc.h>
#include <TNL/Exceptions/BackendSupportMissing.h>
#include <TNL/Backend/Macros.h>

namespace TNL::Allocators {

/**
 * \brief Allocator for page-locked memory on the host.
 *
 * The allocation is done using the `hipHostMalloc` function and the
 * deallocation is done using the `hipHostFree` function.
 *
 * Note: `hipMallocHost` is deprecated and `hipHostMalloc` corresponds
 * to the `cudaMallocHost` function.
 */
template< class T >
struct HipHost
{
   using value_type = T;
   using size_type = std::size_t;
   using difference_type = std::ptrdiff_t;

   HipHost() = default;
   HipHost( const HipHost& ) = default;
   HipHost( HipHost&& ) noexcept = default;

   HipHost&
   operator=( const HipHost& ) = default;
   HipHost&
   operator=( HipHost&& ) noexcept = default;

   template< class U >
   HipHost( const HipHost< U >& )
   {}

   template< class U >
   HipHost( HipHost< U >&& )
   {}

   template< class U >
   HipHost&
   operator=( const HipHost< U >& )
   {
      return *this;
   }

   template< class U >
   HipHost&
   operator=( HipHost< U >&& )
   {
      return *this;
   }

   [[nodiscard]] value_type*
   allocate( size_type n )
   {
#ifdef __HIP__
      value_type* result = nullptr;
      if( hipHostMalloc( (void**) &result, n * sizeof( value_type ), hipHostMallocPortable | hipHostMallocMapped )
          != hipSuccess )
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
      TNL_BACKEND_SAFE_CALL( hipHostFree( (void*) ptr ) );
#else
      throw Exceptions::BackendSupportMissing();
#endif
   }
};

template< class T1, class T2 >
[[nodiscard]] bool
operator==( const HipHost< T1 >&, const HipHost< T2 >& )
{
   return true;
}

template< class T1, class T2 >
[[nodiscard]] bool
operator!=( const HipHost< T1 >& lhs, const HipHost< T2 >& rhs )
{
   return ! ( lhs == rhs );
}

}  // namespace TNL::Allocators
