// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/MemoryOperations.h>

namespace TNL::Algorithms {

template< typename Element, typename Index >
void
MemoryOperations< Devices::Host >::construct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   auto kernel = [ data ]( Index i )
   {
      // placement-new
      ::new( (void*) ( data + i ) ) Element();
   };
   parallelFor< Devices::Host >( 0, size, kernel );
}

template< typename Element, typename Index, typename... Args >
void
MemoryOperations< Devices::Host >::construct( Element* data, Index size, const Args&... args )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   auto kernel = [ data, &args... ]( Index i )
   {
      // placement-new
      // (note that args are passed by reference to the constructor, not via
      // std::forward since move-semantics does not apply for the construction
      // of multiple elements)
      ::new( (void*) ( data + i ) ) Element( args... );
   };
   parallelFor< Devices::Host >( 0, size, kernel );
}

template< typename Element, typename Index >
void
MemoryOperations< Devices::Host >::destruct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to destroy data through a nullptr." );
   auto kernel = [ data ]( Index i )
   {
      ( data + i )->~Element();
   };
   parallelFor< Devices::Host >( 0, size, kernel );
}

template< typename Element >
__cuda_callable__  // only to avoid nvcc warning
void
MemoryOperations< Devices::Host >::setElement( Element* data, const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   *data = value;
}

template< typename Element >
__cuda_callable__  // only to avoid nvcc warning
Element
MemoryOperations< Devices::Host >::getElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
   return *data;
}

}  // namespace TNL::Algorithms
