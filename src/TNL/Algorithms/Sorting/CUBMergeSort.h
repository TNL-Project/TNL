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

/**
 * \brief CUDA merge sort using CUB's DeviceMergeSort.
 *
 * This class provides a wrapper for merge sort implementation from NVIDIA's CUB library.
 * It uses the `cub::DeviceMergeSort::SortKeys` algorithm which performs a merge sort
 * on device-resident data.
 *
 * Merge sort differs from radix sort in several key aspects:
 * - Can handle arbitrary types (as long as they are *LessThan Comparable*) and custom
 *   comparison functors
 * - Is not guaranteed to be stable
 * - Typically slower than DeviceRadixSort for arithmetic types
 *
 * \par Example
 * \include Algorithms/SortingCUBMergeSort.cu
 * \par Output
 * \include SortingCUBMergeSort.out
 *
 * \see CUBRadixSort for a faster radix sort alternative (for arithmetic types)
 * \see https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceMergeSort.html
 */
struct CUBMergeSort
{
   /**
    * \brief Sort array in ascending order.
    *
    * \tparam Array is a type of container to be sorted. It must be
    *         \ref TNL::Containers::Array or \ref TNL::Containers::ArrayView with
    *         \ref TNL::Devices::Cuda device type.
    * \param array The array to sort (will be modified in-place).
    */
   template< typename Array >
   static void
   sort( Array& array )
   {
      sort( array, std::less<>{} );
   }

   /**
    * \brief Sort array using custom comparison function.
    *
    * \param array The array to sort (will be modified in-place).
    * \param compare Comparison function object that returns true if the first
    *        argument should be ordered before the second.
    */
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

      TNL_BACKEND_SAFE_CALL(
         cub::DeviceMergeSort::SortKeys(
            static_cast< void* >( temp_storage.getData() ), temp_storage_bytes, data, size, compare ) );
#else
      throw Exceptions::NotImplementedError( "CUBMergeSort is supported only when CUDA is enabled." );
#endif
   }
};

}  // namespace TNL::Algorithms::Sorting
