// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

#include <TNL/Backend/Macros.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Exceptions/NotImplementedError.h>

#if defined( __HIPCC__ )
   #include <rocprim/device/device_merge_sort.hpp>
#endif

namespace TNL::Algorithms::Sorting {

/**
 * \brief HIP merge sort using rocPRIM's device-wide merge sort.
 *
 * This class wraps `rocprim::merge_sort` for device-resident values. It
 * provides the same default ascending order and custom-comparison interface as
 * \ref CUBMergeSort. The sorted output is produced in an auxiliary array and
 * assigned back to the input container.
 *
 * \par Space requirements
 *
 * The implementation requires an auxiliary array with \f$N\f$ values and
 * rocPRIM's internal temporary storage. Both can be retained in a \ref
 * Workspace and reused by repeated sorts.
 *
 * \see RocPrimRadixSort for a radix sort alternative for arithmetic types.
 * \see
 * https://rocm.docs.amd.com/projects/rocPRIM/en/latest/reference/device.html
 */
struct RocPrimMergeSort
{
   /**
    * \brief Reusable temporary buffers for rocPRIM merge sort.
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
    * \brief Sorts an array into ascending order.
    *
    * \tparam Array is a container with \ref TNL::Devices::GPU device type.
    * \param array is the array or array view to be sorted in place.
    */
   template< typename Array >
   static void
   sort( Array& array )
   {
      sort( array, std::less<>{} );
   }

   /**
    * \brief Sorts an array using a custom comparison function.
    *
    * \tparam Array is a container with \ref TNL::Devices::GPU device type.
    * \tparam Compare is the comparison function object type.
    * \param array is the array or array view to be sorted in place.
    * \param compare returns true if its first argument should be ordered before
    * its second argument.
    */
   template< typename Array, typename Compare >
   static void
   sort( Array& array, const Compare& compare )
   {
      using Device = typename Array::DeviceType;
      static_assert( std::is_same_v< Device, Devices::GPU >, "RocPrimMergeSort: unsupported device type" );

      Workspace< Array > workspace;
      sort( array, workspace, compare );
   }

   /**
    * \brief Sorts an array into ascending order using reusable temporary
    * buffers.
    *
    * \tparam Array is a container with \ref TNL::Devices::GPU device type.
    * \param array is the array or array view to be sorted in place.
    * \param workspace is the scratch space resized on demand and retained for
    * subsequent calls.
    */
   template< typename Array >
   static void
   sort( Array& array, Workspace< Array >& workspace )
   {
      sort( array, workspace, std::less<>{} );
   }

   /**
    * \brief Sorts an array with a custom comparison function and reusable
    * temporary buffers.
    *
    * \tparam Array is a container with \ref TNL::Devices::GPU device type.
    * \tparam Compare is the comparison function object type.
    * \param array is the array or array view to be sorted in place.
    * \param workspace is the scratch space resized on demand and retained for
    * subsequent calls.
    * \param compare returns true if its first argument should be ordered before
    * its second argument.
    */
   template< typename Array, typename Compare >
   static void
   sort( Array& array, Workspace< Array >& workspace, const Compare& compare )
   {
      using Device = typename Array::DeviceType;
      static_assert( std::is_same_v< Device, Devices::GPU >, "RocPrimMergeSort: unsupported device type" );

#if defined( __HIPCC__ )
      const auto size = array.getSize();
      if( size <= 1 )
         return;

      using Value = typename Array::ValueType;
      Value* data = array.getData();
      workspace.auxiliary.setSize( size );

      if( workspace.size != static_cast< std::size_t >( size ) ) {
         workspace.temporaryStorageBytes = 0;
         TNL_BACKEND_SAFE_CALL(
            rocprim::merge_sort(
               nullptr, workspace.temporaryStorageBytes, data, workspace.auxiliary.getData(), size, compare ) );
         workspace.temporaryStorage.setSize( workspace.temporaryStorageBytes );
         workspace.size = size;
      }

      TNL_BACKEND_SAFE_CALL(
         rocprim::merge_sort(
            static_cast< void* >( workspace.temporaryStorage.getData() ),
            workspace.temporaryStorageBytes,
            data,
            workspace.auxiliary.getData(),
            size,
            compare ) );

      array = workspace.auxiliary;
#else
      throw Exceptions::NotImplementedError( "RocPrimMergeSort is supported only when HIP is enabled." );
#endif
   }
};

}  // namespace TNL::Algorithms::Sorting
