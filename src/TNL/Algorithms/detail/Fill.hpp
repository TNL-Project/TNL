// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "Fill.h"

namespace TNL::Algorithms::detail {

template< typename Element, typename Index >
__cuda_callable__
void
Fill< Devices::Sequential >::fill( Element* data, const Element& value, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   for( Index i = 0; i < size; i++ )
      data[ i ] = value;
}

template< typename Element, typename Index >
void
Fill< Devices::Host >::fill( Element* data, const Element& value, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   auto kernel = [ data, value ]( Index i )
   {
      data[ i ] = value;
   };
   parallelFor< Devices::Host >( 0, size, kernel );
}

template< typename Element, typename Index >
void
Fill< Devices::Cuda >::fill( Element* data, const Element& value, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   auto kernel = [ data, value ] __cuda_callable__( Index i )
   {
      data[ i ] = value;
   };
   parallelFor< Devices::Cuda >( 0, size, kernel );
}

}  // namespace TNL::Algorithms::detail
