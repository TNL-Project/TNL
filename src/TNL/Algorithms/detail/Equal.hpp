// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>  // std::equal

#include <TNL/Algorithms/reduce.h>
#include <TNL/Exceptions/CudaSupportMissing.h>

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
   /***
    * Here, destination is on host and source is on CUDA device.
    */
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
#ifdef __CUDACC__
   const int buffer_size = TNL::min( Cuda::getTransferBufferSize() / sizeof( Element2 ), size );
   std::unique_ptr< Element2[] > host_buffer{ new Element2[ buffer_size ] };
   Index compared = 0;
   while( compared < size ) {
      const int transfer = TNL::min( size - compared, buffer_size );
      if( cudaMemcpy(
             (void*) host_buffer.get(), (void*) &source[ compared ], transfer * sizeof( Element2 ), cudaMemcpyDeviceToHost )
          != cudaSuccess )
         std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
      if( ! Equal< Devices::Host >::equal( &destination[ compared ], host_buffer.get(), transfer ) )
         return false;
      compared += transfer;
   }
   return true;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename DeviceType >
template< typename Element1, typename Element2, typename Index >
bool
Equal< Devices::Cuda, DeviceType >::equal( const Element1* destination, const Element2* source, Index size )
{
   return Equal< DeviceType, Devices::Cuda >::equal( source, destination, size );
}

}  // namespace TNL::Algorithms::detail
