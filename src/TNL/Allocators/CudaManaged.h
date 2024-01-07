// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Exceptions/BackendBadAlloc.h>
#include <TNL/Exceptions/BackendSupportMissing.h>
#include <TNL/Backend/Macros.h>

namespace TNL::Allocators {

/**
 * \brief Allocator for the CUDA Unified Memory system.
 *
 * The memory allocated by this allocator will be automatically managed by the
 * CUDA Unified Memory system. The allocation is done using the
 * `cudaMallocManaged` function and the deallocation is done using the
 * `cudaFree` function.
 */
template< class T >
struct CudaManaged
{
   using value_type = T;
   using size_type = std::size_t;
   using difference_type = std::ptrdiff_t;

   CudaManaged() = default;
   CudaManaged( const CudaManaged& ) = default;
   CudaManaged( CudaManaged&& ) noexcept = default;

   CudaManaged&
   operator=( const CudaManaged& ) = default;
   CudaManaged&
   operator=( CudaManaged&& ) noexcept = default;

   template< class U >
   CudaManaged( const CudaManaged< U >& )
   {}

   template< class U >
   CudaManaged( CudaManaged< U >&& )
   {}

   template< class U >
   CudaManaged&
   operator=( const CudaManaged< U >& )
   {
      return *this;
   }

   template< class U >
   CudaManaged&
   operator=( CudaManaged< U >&& )
   {
      return *this;
   }

   [[nodiscard]] value_type*
   allocate( size_type n )
   {
#ifdef __CUDACC__
      value_type* result = nullptr;
      if( cudaMallocManaged( &result, n * sizeof( value_type ) ) != cudaSuccess )
         throw Exceptions::BackendBadAlloc();
      return result;
#else
      throw Exceptions::BackendSupportMissing();
#endif
   }

   void
   deallocate( value_type* ptr, size_type )
   {
#ifdef __CUDACC__
      TNL_BACKEND_SAFE_CALL( cudaFree( (void*) ptr ) );
#else
      throw Exceptions::BackendSupportMissing();
#endif
   }
};

template< class T1, class T2 >
[[nodiscard]] bool
operator==( const CudaManaged< T1 >&, const CudaManaged< T2 >& )
{
   return true;
}

template< class T1, class T2 >
[[nodiscard]] bool
operator!=( const CudaManaged< T1 >& lhs, const CudaManaged< T2 >& rhs )
{
   return ! ( lhs == rhs );
}

}  // namespace TNL::Allocators
