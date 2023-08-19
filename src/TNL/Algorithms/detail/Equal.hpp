// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>  // std::equal

#include <TNL/Backend.h>
#include <TNL/Algorithms/reduce.h>

#include "Equal.h"

namespace TNL::Algorithms::detail {

template< typename Element1, typename Element2, typename Index >
__cuda_callable__
bool
Equal< Devices::Sequential >::equal( const Element1* destination, const Element2* source, Index size )
{
   if( size == 0 )
      return true;
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );

   for( Index i = 0; i < size; i++ )
      if( ! ( destination[ i ] == source[ i ] ) )
         return false;
   return true;
}

template< typename Element1, typename Element2, typename Index >
bool
Equal< Devices::Host >::equal( const Element1* destination, const Element2* source, Index size )
{
   if( size == 0 )
      return true;
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );

   if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 ) {
      auto fetch = [ destination, source ]( Index i ) -> bool
      {
         return destination[ i ] == source[ i ];
      };
      return reduce< Devices::Host >( (Index) 0, size, fetch, std::logical_and<>{}, true );
   }
   else {
      // sequential algorithm can return as soon as it finds a mismatch
      return std::equal( source, source + size, destination );
   }
}

template< typename Element1, typename Element2, typename Index >
bool
Equal< Devices::Cuda >::equal( const Element1* destination, const Element2* source, Index size )
{
   if( size == 0 )
      return true;
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );

   auto fetch = [ = ] __cuda_callable__( Index i ) -> bool
   {
      return destination[ i ] == source[ i ];
   };
   return reduce< Devices::Cuda >( (Index) 0, size, fetch, std::logical_and<>{}, true );
}

template< typename DeviceType >
template< typename Element1, typename Element2, typename Index >
bool
Equal< DeviceType, Devices::Cuda >::equal( const Element1* destination, const Element2* source, Index size )
{
   if( size == 0 )
      return true;

   // Here, destination is on host and source is on CUDA device.
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );

   bool result = true;
   auto push = [ &result, destination ]( std::size_t offset, const Element2* buffer, std::size_t buffer_size, bool& next_iter )
   {
      result = next_iter = Equal< Devices::Host >::equal( &destination[ offset ], buffer, buffer_size );
   };
   Backend::bufferedTransferToHost( source, size, push );
   return result;
}

template< typename DeviceType >
template< typename Element1, typename Element2, typename Index >
bool
Equal< Devices::Cuda, DeviceType >::equal( const Element1* destination, const Element2* source, Index size )
{
   return Equal< DeviceType, Devices::Cuda >::equal( source, destination, size );
}

}  // namespace TNL::Algorithms::detail
