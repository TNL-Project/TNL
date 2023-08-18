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
 * \brief Allocator for page-locked memory on the host.
 *
 * The allocation is done using the `cudaMallocHost` function and the
 * deallocation is done using the `cudaFreeHost` function.
 */
template< class T >
struct CudaHost
{
   using value_type = T;
   using size_type = std::size_t;
   using difference_type = std::ptrdiff_t;

   CudaHost() = default;
   CudaHost( const CudaHost& ) = default;
   CudaHost( CudaHost&& ) noexcept = default;

   CudaHost&
   operator=( const CudaHost& ) = default;
   CudaHost&
   operator=( CudaHost&& ) noexcept = default;

   template< class U >
   CudaHost( const CudaHost< U >& )
   {}

   template< class U >
   CudaHost( CudaHost< U >&& )
   {}

   template< class U >
   CudaHost&
   operator=( const CudaHost< U >& )
   {
      return *this;
   }

   template< class U >
   CudaHost&
   operator=( CudaHost< U >&& )
   {
      return *this;
   }

   [[nodiscard]] value_type*
   allocate( size_type n )
   {
#ifdef __CUDACC__
      TNL_CHECK_CUDA_DEVICE;
      value_type* result = nullptr;
      // cudaHostAllocPortable - The memory returned by this call will be considered as pinned memory by all
      //                       CUDA contexts, not just the one that performed the allocation.
      // cudaHostAllocMapped - Maps the allocation into the CUDA address space.
      // Also note that we assume that the cudaDevAttrCanUseHostPointerForRegisteredMem attribute is non-zero
      // on all devices visible to the application, in which case the pointer returned by cudaMallocHost can
      // be used directly by all devices without having to call cudaHostGetDevicePointer. See the reference:
      // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc00502b44e5f1bdc0b424487ebb08db0
      if( cudaMallocHost( (void**) &result, n * sizeof( value_type ), cudaHostAllocPortable | cudaHostAllocMapped )
          != cudaSuccess )
         throw Exceptions::BackendBadAlloc();
      TNL_CHECK_CUDA_DEVICE;
      return result;
#else
      throw Exceptions::BackendSupportMissing();
#endif
   }

   void
   deallocate( value_type* ptr, size_type )
   {
#ifdef __CUDACC__
      TNL_CHECK_CUDA_DEVICE;
      cudaFreeHost( (void*) ptr );
      TNL_CHECK_CUDA_DEVICE;
#else
      throw Exceptions::BackendSupportMissing();
#endif
   }
};

template< class T1, class T2 >
[[nodiscard]] bool
operator==( const CudaHost< T1 >&, const CudaHost< T2 >& )
{
   return true;
}

template< class T1, class T2 >
[[nodiscard]] bool
operator!=( const CudaHost< T1 >& lhs, const CudaHost< T2 >& rhs )
{
   return ! ( lhs == rhs );
}

}  // namespace TNL::Allocators
