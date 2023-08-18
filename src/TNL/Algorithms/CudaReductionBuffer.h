// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>  // std::byte

#include <TNL/Assert.h>
#include <TNL/Allocators/Default.h>

namespace TNL::Algorithms {

class CudaReductionBuffer
{
public:
   static CudaReductionBuffer&
   getInstance()
   {
      // note that this ensures construction on first use, and thus also correct
      // destruction before the CUDA context is destroyed
      // https://stackoverflow.com/questions/335369/finding-c-static-initialization-order-problems#335746
      static CudaReductionBuffer instance;
      return instance;
   }

   void
   setSize( std::size_t size )
   {
      if( size > this->size ) {
         this->reset();
         this->data = allocator.allocate( size );
         this->size = size;
      }
   }

   void
   reset()
   {
      if( data != nullptr )
         allocator.deallocate( data, size );
      data = nullptr;
      size = 0;
   }

   template< typename Type >
   Type*
   getData()
   {
      return reinterpret_cast< Type* >( this->data );
   }

   ~CudaReductionBuffer()
   {
      reset();
   }

   // copy-constructor and copy-assignment are meaningless for a singleton class
   CudaReductionBuffer( CudaReductionBuffer const& copy ) = delete;
   CudaReductionBuffer&
   operator=( CudaReductionBuffer const& copy ) = delete;

private:
   // private constructor of the singleton
   CudaReductionBuffer( std::size_t size = 0 )
   {
      setSize( size );
   }

   std::byte* data = nullptr;

   std::size_t size = 0;

   using Allocator = Allocators::Default< Devices::Cuda >::template Allocator< std::byte >;
   Allocator allocator;
};

}  // namespace TNL::Algorithms
