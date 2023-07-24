// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/MemoryOperations.h>
#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/parallelFor.h>

namespace TNL::Algorithms {

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
   parallelFor< Devices::Cuda >( 0, size, kernel );
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
   parallelFor< Devices::Cuda >( 0, size, kernel, args... );
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
   parallelFor< Devices::Cuda >( 0, size, kernel );
}

template< typename Element >
__cuda_callable__
void
MemoryOperations< Devices::Cuda >::setElement( Element* data, const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
#ifdef __CUDA_ARCH__
   *data = value;
#else
   // NOTE: calling `MemoryOperations< Devices::Cuda >::set( data, value, 1 );`
   // does not work here due to `#ifdef __CUDA_ARCH__` above. It would involve
   // launching a CUDA kernel with an extended lambda, which would be discarded
   // by nvcc (never called).
   copy< Devices::Cuda, void >( data, &value, 1 );
#endif
}

template< typename Element >
__cuda_callable__
Element
MemoryOperations< Devices::Cuda >::getElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
#ifdef __CUDA_ARCH__
   return *data;
#else
   Element result;
   copy< void, Devices::Cuda >( &result, data, 1 );
   return result;
#endif
}

}  // namespace TNL::Algorithms
