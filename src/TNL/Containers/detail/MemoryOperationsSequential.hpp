// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "MemoryOperations.h"

namespace TNL::Containers::detail {

template< typename Element, typename Index >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::construct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   for( Index i = 0; i < size; i++ )
      // placement-new
      ::new( (void*) ( data + i ) ) Element();
}

template< typename Element, typename Index, typename... Args >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::construct( Element* data, Index size, const Args&... args )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   for( Index i = 0; i < size; i++ )
      // placement-new
      // (note that args are passed by reference to the constructor, not via
      // std::forward since move-semantics does not apply for the construction
      // of multiple elements)
      ::new( (void*) ( data + i ) ) Element( args... );
}

template< typename Element, typename Index >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::destruct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to destroy elements through a nullptr." );
   for( Index i = 0; i < size; i++ )
      ( data + i )->~Element();
}

template< typename Element >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::setElement( Element* data, const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   *data = value;
}

template< typename Element >
__cuda_callable__
Element
MemoryOperations< Devices::Sequential >::getElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
   return *data;
}

}  // namespace TNL::Containers::detail
