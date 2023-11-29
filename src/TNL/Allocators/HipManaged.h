// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Exceptions/BackendBadAlloc.h>
#include <TNL/Exceptions/BackendSupportMissing.h>
#include <TNL/Backend/Macros.h>

namespace TNL::Allocators {

/**
 * \brief Allocator for the HIP Unified Memory system.
 *
 * The memory allocated by this allocator will be automatically managed by the
 * HIP Unified Memory system. The allocation is done using the
 * `hipMallocManaged` function and the deallocation is done using the
 * `hipFree` function.
 */
template< class T >
struct HipManaged
{
   using value_type = T;
   using size_type = std::size_t;
   using difference_type = std::ptrdiff_t;

   HipManaged() = default;
   HipManaged( const HipManaged& ) = default;
   HipManaged( HipManaged&& ) noexcept = default;

   HipManaged&
   operator=( const HipManaged& ) = default;
   HipManaged&
   operator=( HipManaged&& ) noexcept = default;

   template< class U >
   HipManaged( const HipManaged< U >& )
   {}

   template< class U >
   HipManaged( HipManaged< U >&& )
   {}

   template< class U >
   HipManaged&
   operator=( const HipManaged< U >& )
   {
      return *this;
   }

   template< class U >
   HipManaged&
   operator=( HipManaged< U >&& )
   {
      return *this;
   }

   [[nodiscard]] value_type*
   allocate( size_type n )
   {
#ifdef __HIP__
      value_type* result = nullptr;
      if( hipMallocManaged( &result, n * sizeof( value_type ) ) != hipSuccess )
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
operator==( const HipManaged< T1 >&, const HipManaged< T2 >& )
{
   return true;
}

template< class T1, class T2 >
[[nodiscard]] bool
operator!=( const HipManaged< T1 >& lhs, const HipManaged< T2 >& rhs )
{
   return ! ( lhs == rhs );
}

}  // namespace TNL::Allocators
