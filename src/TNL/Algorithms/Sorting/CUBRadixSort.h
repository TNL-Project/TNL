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
   #include <cub/device/device_radix_sort.cuh>
#endif

namespace TNL::Algorithms::Sorting {

/**
 * \brief CUDA radix sort using CUB's DeviceRadixSort.
 *
 * This class provides a wrapper for radix sort implementation from NVIDIA's CUB library.
 * It uses the `cub::DeviceRadixSort::SortKeys` algorithm which performs a stable
 * radix sort on device-resident data.
 *
 * \par Stability
 *
 * Radix sort is a stable sorting algorithm.
 *
 * \par Space requirements
 *
 * This implementation requires additional storage:
 * - An auxiliary array of the same size (\f$N\f$) as the input array to temporarily hold
 *   sorted output
 * - Temporary storage for CUB's internal operations (\f$O(N+P)\f$ where P is the number
 *   of streaming multiprocessors, typically a small constant)
 *
 * The total memory overhead is approximately \f$2N + P\f$.
 *
 * \par Example
 * \include Algorithms/SortingCUBRadixSort.cu
 * \par Output
 * \include SortingCUBRadixSort.out
 *
 * \see CUBMergeSort for a general merge sort alternative (for arbitrary types and comparison functors)
 * \see https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html
 */
struct CUBRadixSort
{
   /**
    * \brief Sort an array of arithmetic values into ascending order.
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
      using Value = typename Array::ValueType;
      using Device = typename Array::DeviceType;
      static_assert( std::is_arithmetic_v< Value >, "CUBRadixSort: unsupported value type" );
      static_assert( std::is_same_v< Device, Devices::Cuda >, "CUBRadixSort: unsupported device type" );

#if defined( __CUDACC__ )
      const auto size = array.getSize();
      if( size <= 1 )
         return;

      Value* data = array.getData();

      Containers::Array< Value, Devices::Cuda > temp_array;
      temp_array.setSize( size );

      std::size_t temp_storage_bytes = 0;
      TNL_BACKEND_SAFE_CALL( cub::DeviceRadixSort::SortKeys( nullptr, temp_storage_bytes, data, temp_array.getData(), size ) );

      Containers::Array< std::uint8_t, Devices::Cuda > temp_storage;
      temp_storage.setSize( temp_storage_bytes );

      TNL_BACKEND_SAFE_CALL(
         cub::DeviceRadixSort::SortKeys(
            static_cast< void* >( temp_storage.getData() ), temp_storage_bytes, data, temp_array.getData(), size ) );

      array = temp_array;
#else
      throw Exceptions::NotImplementedError( "CUBRadixSort is supported only when CUDA is enabled." );
#endif
   }
};

}  // namespace TNL::Algorithms::Sorting
