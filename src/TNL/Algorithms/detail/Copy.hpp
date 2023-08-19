// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>    // std::copy
#include <stdexcept>    // std::length_error
#include <type_traits>  // std::remove_cv_t

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Algorithms/parallelFor.h>

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
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );

   for( Index i = 0; i < size; i++ )
      destination[ i ] = source[ i ];
}

template< typename DestinationElement, typename Index, typename SourceIterator >
void
Copy< Devices::Sequential >::copy( DestinationElement* destination,
                                   Index destinationSize,
                                   SourceIterator begin,
                                   SourceIterator end )
{
   Index i = 0;
   while( i < destinationSize && begin != end )
      destination[ i++ ] = *begin++;
   if( begin != end )
      throw std::length_error( "Source iterator is larger than the destination array." );
}

template< typename DestinationElement, typename SourceElement, typename Index >
void
Copy< Devices::Host >::copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );

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

template< typename DestinationElement, typename Index, typename SourceIterator >
void
Copy< Devices::Host >::copy( DestinationElement* destination, Index destinationSize, SourceIterator begin, SourceIterator end )
{
   Copy< Devices::Sequential >::copy( destination, destinationSize, begin, end );
}

template< typename DestinationElement, typename SourceElement, typename Index >
void
Copy< Devices::Cuda >::copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );

   // our ParallelFor kernel is faster than cudaMemcpy
   auto kernel = [ destination, source ] __cuda_callable__( Index i )
   {
      destination[ i ] = source[ i ];
   };
   parallelFor< Devices::Cuda >( 0, size, kernel );
}

template< typename DestinationElement, typename Index, typename SourceIterator >
void
Copy< Devices::Cuda >::copy( DestinationElement* destination, Index destinationSize, SourceIterator begin, SourceIterator end )
{
   TNL_ASSERT_GE( destinationSize, (Index) 0, "Array size must be non-negative." );
   using BaseType = typename std::remove_cv_t< DestinationElement >;
   std::size_t copied_elements = 0;
   auto fill = [ & ]( std::size_t offset, BaseType* buffer, std::size_t buffer_size )
   {
      TNL_ASSERT_LE(
         offset + buffer_size, (std::size_t) destinationSize, "bufferedTransferToDevice supplied wrong offset or buffer size" );
      copied_elements = 0;
      while( copied_elements < buffer_size && begin != end )
         buffer[ copied_elements++ ] = *begin++;
   };
   // NOTE: capture by reference is needed for copied_elements to get its updated values
   auto push =
      [ &copied_elements, destination ]( std::size_t offset, const BaseType* buffer, std::size_t buffer_size, bool& next_iter )
   {
      Copy< Devices::Cuda, Devices::Sequential >::copy( destination + offset, buffer, copied_elements );
      next_iter = copied_elements == buffer_size;
   };
   Backend::bufferedTransfer< BaseType >( destinationSize, fill, push );
   if( begin != end )
      throw std::length_error( "Source iterator is larger than the destination array." );
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
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );

   if constexpr( std::is_same_v< std::remove_cv_t< DestinationElement >, std::remove_cv_t< SourceElement > > ) {
      Backend::memcpy( destination, source, size * sizeof( DestinationElement ), Backend::MemcpyDeviceToHost );
   }
   else {
      auto push = [ = ]( std::size_t offset, const SourceElement* buffer, std::size_t buffer_size, bool& next_iter )
      {
         TNL_ASSERT_LE(
            offset + buffer_size, (std::size_t) size, "bufferedTransferToHost supplied wrong offset or buffer size" );
         for( std::size_t i = 0; i < buffer_size; i++ )
            destination[ i + offset ] = buffer[ i ];
      };
      Backend::bufferedTransferToHost( source, size, push );
   }
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

   if constexpr( std::is_same_v< std::remove_cv_t< DestinationElement >, std::remove_cv_t< SourceElement > > ) {
      Backend::memcpy( destination, source, size * sizeof( DestinationElement ), Backend::MemcpyHostToDevice );
   }
   else {
      auto fill = [ = ]( std::size_t offset, DestinationElement* buffer, std::size_t buffer_size )
      {
         TNL_ASSERT_LE(
            offset + buffer_size, (std::size_t) size, "bufferedTransferToDevice supplied wrong offset or buffer size" );
         for( std::size_t i = 0; i < buffer_size; i++ )
            buffer[ i ] = source[ i + offset ];
      };
      Backend::bufferedTransferToDevice( destination, size, fill );
   }
}

}  // namespace TNL::Algorithms::detail
