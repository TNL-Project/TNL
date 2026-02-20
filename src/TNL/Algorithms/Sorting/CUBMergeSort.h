// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <TNL/Backend/Macros.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Exceptions/NotImplementedError.h>

#if defined( __CUDACC__ )
   #include <cub/device/device_merge_sort.cuh>
#endif

namespace TNL::Algorithms::Sorting {

struct CUBMergeSort
{
   template< typename Array >
   static void
   sort( Array& array )
   {
      using Value = typename Array::ValueType;
      sort( array,
            [] __cuda_callable__( const Value& a, const Value& b )
            {
               return a < b;
            } );
   }

   template< typename Array, typename Compare >
   static void
   sort( Array& array, const Compare& compare )
   {
      using Device = typename Array::DeviceType;
      static_assert( std::is_same_v< Device, Devices::Cuda >, "CUBMergeSort: unsupported device type" );
#if defined( __CUDACC__ )
      const auto size = array.getSize();
      if( size <= 1 )
         return;

      using Value = typename Array::ValueType;
      Value* data = array.getData();

      std::size_t temp_storage_bytes = 0;
      TNL_BACKEND_SAFE_CALL( cub::DeviceMergeSort::SortKeys( nullptr, temp_storage_bytes, data, size, compare ) );

      Containers::Array< std::uint8_t, Devices::Cuda > temp_storage;
      temp_storage.setSize( temp_storage_bytes );

      TNL_BACKEND_SAFE_CALL( cub::DeviceMergeSort::SortKeys(
         static_cast< void* >( temp_storage.getData() ), temp_storage_bytes, data, size, compare ) );
#else
      throw Exceptions::NotImplementedError( "CUBMergeSort is supported only when CUDA is enabled." );
#endif
   }
};

}  // namespace TNL::Algorithms::Sorting
