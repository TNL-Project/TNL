// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <TNL/Backend/Macros.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Exceptions/NotImplementedError.h>

#if defined( __HIPCC__ )
   #include <rocprim/device/device_radix_sort.hpp>
#endif

namespace TNL::Algorithms::Sorting {

/**
 * \brief HIP radix sort using rocPRIM's device-wide radix sort.
 *
 * This class wraps `rocprim::radix_sort_keys`, which performs a stable
 * ascending radix sort on device-resident arithmetic values. The wrapper uses
 * an auxiliary array for the sorted output and assigns it back to the input
 * container. As with \ref CUBRadixSort, passing an
 * \ref TNL::Containers::ArrayView performs a device-to-device copy-back, while
 * assigning an owning
 * \ref TNL::Containers::Array replaces its storage.
 *
 * \par Space requirements
 *
 * The implementation requires an auxiliary array with \f$N\f$ values and
 * rocPRIM's internal temporary storage. Both can be retained in a \ref
 * Workspace and reused by repeated sorts.
 *
 * \see RocPrimMergeSort for a comparison-based alternative.
 * \see
 * https://rocm.docs.amd.com/projects/rocPRIM/en/latest/reference/device.html
 */
struct RocPrimRadixSort
{
   /**
    * \brief Reusable temporary buffers for rocPRIM radix sort.
    *
    * Repeated sorts should pass array views with the same workspace to avoid
    * device memory allocation in the sort calls. The buffers are resized when
    * the array size changes.
    *
    * \tparam Array is the container type to be sorted.
    */
   template< typename Array >
   struct Workspace
   {
      //! \brief Auxiliary array receiving the sorted values.
      Containers::Array< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType > auxiliary;

      //! \brief Device storage used internally by rocPRIM.
      Containers::Array< std::uint8_t, typename Array::DeviceType, std::size_t > temporaryStorage;

      //! \brief Required size of the internal temporary storage in bytes.
      std::size_t temporaryStorageBytes = 0;

      //! \brief Array size for which the temporary storage was queried.
      std::size_t size = 0;
   };

   /**
    * \brief Sorts an array of arithmetic values into ascending order.
    *
    * \tparam Array is a container with \ref TNL::Devices::GPU device type and an
    * arithmetic value type.
    * \param array is the array or array view to be sorted in place.
    */
   template< typename Array >
   static void
   sort( Array& array )
   {
      using Value = typename Array::ValueType;
      using Device = typename Array::DeviceType;
      static_assert( std::is_arithmetic_v< Value >, "RocPrimRadixSort: unsupported value type" );
      static_assert( std::is_same_v< Device, Devices::GPU >, "RocPrimRadixSort: unsupported device type" );

      Workspace< Array > workspace;
      sort( array, workspace );
   }

   /**
    * \brief Sorts an array into ascending order using reusable temporary
    * buffers.
    *
    * \tparam Array is a container with \ref TNL::Devices::GPU device type and an
    * arithmetic value type.
    * \param array is the array or array view to be sorted in place.
    * \param workspace is the scratch space resized on demand and retained for
    * subsequent calls.
    */
   template< typename Array >
   static void
   sort( Array& array, Workspace< Array >& workspace )
   {
      using Value = typename Array::ValueType;
      using Device = typename Array::DeviceType;
      static_assert( std::is_arithmetic_v< Value >, "RocPrimRadixSort: unsupported value type" );
      static_assert( std::is_same_v< Device, Devices::GPU >, "RocPrimRadixSort: unsupported device type" );

#if defined( __HIPCC__ )
      const auto size = array.getSize();
      if( size <= 1 )
         return;

      Value* data = array.getData();
      workspace.auxiliary.setSize( size );

      if( workspace.size != static_cast< std::size_t >( size ) ) {
         workspace.temporaryStorageBytes = 0;
         TNL_BACKEND_SAFE_CALL(
            rocprim::radix_sort_keys( nullptr, workspace.temporaryStorageBytes, data, workspace.auxiliary.getData(), size ) );
         workspace.temporaryStorage.setSize( workspace.temporaryStorageBytes );
         workspace.size = size;
      }

      TNL_BACKEND_SAFE_CALL(
         rocprim::radix_sort_keys(
            static_cast< void* >( workspace.temporaryStorage.getData() ),
            workspace.temporaryStorageBytes,
            data,
            workspace.auxiliary.getData(),
            size ) );

      array = workspace.auxiliary;
#else
      throw Exceptions::NotImplementedError( "RocPrimRadixSort is supported only when HIP is enabled." );
#endif
   }
};

}  // namespace TNL::Algorithms::Sorting
