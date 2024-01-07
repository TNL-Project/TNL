// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/parallelFor.h>

#include "MemoryOperations.h"

namespace TNL::Containers::detail {

template< typename Element, typename Index >
void
MemoryOperations< Devices::Cuda >::construct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   auto kernel = [ data ] __cuda_callable__( Index i )
   {
      // placement-new
      ::new( (void*) ( data + i ) ) Element();
   };
   Algorithms::parallelFor< Devices::Cuda >( 0, size, kernel );
}

template< typename Element, typename Index, typename... Args >
void
MemoryOperations< Devices::Cuda >::construct( Element* data, Index size, const Args&... args )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   // NOTE: nvcc does not allow __cuda_callable__ lambdas with a variadic capture
   auto kernel = [ data ] __cuda_callable__( Index i, Args... args )
   {
      // placement-new
      // (note that args are passed by value to the constructor, not via
      // std::forward or even by reference, since move-semantics does not apply for
      // the construction of multiple elements and pass-by-reference cannot be used
      // with CUDA kernels)
      ::new( (void*) ( data + i ) ) Element( args... );
   };
   Algorithms::parallelFor< Devices::Cuda >( 0, size, kernel, args... );
}

template< typename Element, typename Index >
void
MemoryOperations< Devices::Cuda >::destruct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to destroy data through a nullptr." );
   auto kernel = [ data ] __cuda_callable__( Index i )
   {
      ( data + i )->~Element();
   };
   Algorithms::parallelFor< Devices::Cuda >( 0, size, kernel );
}

template< typename Element >
__cuda_callable__
void
MemoryOperations< Devices::Cuda >::setElement( Element* data, const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   *data = value;
#else
   Algorithms::copy< Devices::Cuda, void >( data, &value, 1 );
#endif
}

template< typename Element >
__cuda_callable__
Element
MemoryOperations< Devices::Cuda >::getElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   return *data;
#else
   Element result;
   Algorithms::copy< void, Devices::Cuda >( &result, data, 1 );
   return result;
#endif
}

}  // namespace TNL::Containers::detail
