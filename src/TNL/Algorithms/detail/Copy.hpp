// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>    // std::copy
#include <memory>       // std::unique_ptr
#include <type_traits>  // std::remove_cv_t

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Exceptions/CudaSupportMissing.h>

#include "Copy.h"

namespace TNL::Algorithms::detail {

template< typename DestinationElement, typename SourceElement, typename Index >
__cuda_callable__
void
Copy< Devices::Sequential >::copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );

   for( Index i = 0; i < size; i++ )
      destination[ i ] = source[ i ];
}

template< typename DestinationElement, typename SourceElement, typename Index >
void
Copy< Devices::Host >::copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );

   // our ParallelFor version is faster than std::copy iff we use more than 1 thread
   if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 ) {
      auto kernel = [ destination, source ]( Index i )
      {
         destination[ i ] = source[ i ];
      };
      parallelFor< Devices::Host >( 0, size, kernel );
   }
   else {
      // std::copy usually uses std::memcpy for TriviallyCopyable types
      std::copy( source, source + size, destination );
   }
}

template< typename DestinationElement, typename SourceElement, typename Index >
void
Copy< Devices::Cuda >::copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );

   // our ParallelFor kernel is faster than cudaMemcpy
   auto kernel = [ destination, source ] __cuda_callable__( Index i )
   {
      destination[ i ] = source[ i ];
   };
   parallelFor< Devices::Cuda >( 0, size, kernel );
}

template< typename DeviceType >
template< typename DestinationElement, typename SourceElement, typename Index >
void
Copy< DeviceType, Devices::Cuda >::copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
#ifdef __CUDACC__
   if( std::is_same< std::remove_cv_t< DestinationElement >, std::remove_cv_t< SourceElement > >::value ) {
      if( cudaMemcpy( destination, source, size * sizeof( DestinationElement ), cudaMemcpyDeviceToHost ) != cudaSuccess )
         std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
   }
   else {
      using BaseType = std::remove_cv_t< SourceElement >;
      const int buffer_size = TNL::min( Cuda::getTransferBufferSize() / sizeof( BaseType ), size );
      std::unique_ptr< BaseType[] > buffer{ new BaseType[ buffer_size ] };
      Index i = 0;
      while( i < size ) {
         if( cudaMemcpy( (void*) buffer.get(),
                         (void*) &source[ i ],
                         TNL::min( size - i, buffer_size ) * sizeof( SourceElement ),
                         cudaMemcpyDeviceToHost )
             != cudaSuccess )
            std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
         TNL_CHECK_CUDA_DEVICE;
         int j = 0;
         while( j < buffer_size && i + j < size ) {
            destination[ i + j ] = buffer[ j ];
            j++;
         }
         i += j;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename DeviceType >
template< typename DestinationElement, typename SourceElement, typename Index >
void
Copy< Devices::Cuda, DeviceType >::copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
#ifdef __CUDACC__
   if( std::is_same< std::remove_cv_t< DestinationElement >, std::remove_cv_t< SourceElement > >::value ) {
      if( cudaMemcpy( destination, source, size * sizeof( DestinationElement ), cudaMemcpyHostToDevice ) != cudaSuccess )
         std::cerr << "Transfer of data from host to CUDA device failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
   }
   else {
      const int buffer_size = TNL::min( Cuda::getTransferBufferSize() / sizeof( DestinationElement ), size );
      std::unique_ptr< DestinationElement[] > buffer{ new DestinationElement[ buffer_size ] };
      Index i = 0;
      while( i < size ) {
         int j = 0;
         while( j < buffer_size && i + j < size ) {
            buffer[ j ] = source[ i + j ];
            j++;
         }
         if( cudaMemcpy(
                (void*) &destination[ i ], (void*) buffer.get(), j * sizeof( DestinationElement ), cudaMemcpyHostToDevice )
             != cudaSuccess )
            std::cerr << "Transfer of data from host to CUDA device failed." << std::endl;
         TNL_CHECK_CUDA_DEVICE;
         i += j;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

}  // namespace TNL::Algorithms::detail
