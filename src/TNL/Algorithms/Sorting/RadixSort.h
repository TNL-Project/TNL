// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <type_traits>

#include <TNL/Algorithms/Sorting/detail/radixSort.h>

namespace TNL::Algorithms::Sorting {

/**
 * \brief Least significant digit (LSD) radix sort for GPU.
 *
 * This is a native GPU implementation processing 8-bit digits of an
 * order-preserving sort key, one digit per pass. Each pass builds per-block
 * histograms of digit counts, exclusive-scans them, and scatters the
 * elements to an auxiliary array. Unlike \ref CUBRadixSort, the value type
 * does not need to be directly supported by CUB, but only arithmetic types
 * can be sorted because the key transformation relies on their bit layout.
 *
 * \par Stability
 *
 * Radix sort is a stable sorting algorithm.
 *
 * \par Space requirements
 *
 * The sort requires an auxiliary array of the same size (\f$N\f$) as the
 * input array, which serves as the ping-pong buffer of the passes, and an
 * array of tile status counters of the decoupled lookback, which is at most
 * \f$2^{17}\f$ tiles times 256 counters (about 128 MiB). When a workspace
 * is reused across sorts, both buffers are persistent and only grow.
 *
 * \par Example
 *
 * \include Algorithms/SortingRadixSort.cu
 * \par Output
 *
 * \include SortingRadixSort.out
 *
 * \see CUBRadixSort for a wrapper of the radix sort from the CUB library
 * \see CUBMergeSort for a comparison-based alternative (for arbitrary types
 * and comparison functors)
 */
struct RadixSort
{
   /**
    * \brief Workspace of the temporary device buffers of the radix sort.
    *
    * Repeated sorts of arrays of a similar size should reuse one workspace
    * (see detail::RadixSortWorkspace) to remove the per-sort device memory
    * allocations: the buffers are grown on demand by the `sort` overloads
    * taking the workspace and are never shrunk, so no device memory is
    * allocated once the buffers fit the sorted array. The dominant buffer
    * is the auxiliary (ping-pong) array with one value per element of the
    * sorted array.
    *
    * Arrays of at most one tile (2048 elements) are sorted by a single
    * kernel launch without any temporary buffers, so the workspace is not
    * touched at all at these sizes.
    *
    * \note The sorts sharing one workspace must not overlap: the workspace
    * carries state between asynchronous kernel launches, so the sorts must
    * be ordered on the same stream.
    *
    * \tparam Value is the value type of the arrays to be sorted.
    * \tparam Device is the device type, only \ref Devices::GPU is implemented.
    * \tparam Index is the index type of the arrays to be sorted.
    */
   template< typename Value, typename Device, typename Index >
   using Workspace = detail::RadixSortWorkspace< Value, Device, Index >;

   /**
    * \brief Sorts `array` in ascending order.
    *
    * \tparam Array is the type of container to be sorted. It can be, for
    * example, \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
    * \ref TNL::Containers::Vector or \ref TNL::Containers::VectorView with
    * \ref TNL::Devices::Cuda or \ref TNL::Devices::Hip device type.
    * The value type must be an arithmetic type with at most 8 bytes.
    * \param array is an instance of the container to be sorted in place.
    */
   template< typename Array >
   void static sort( Array& array )
   {
      sort( array, std::less{} );
   }

   /**
    * \brief Sorts `array` in ascending or descending order.
    *
    * Unlike the other sorters, the radix sort cannot use an arbitrary
    * comparison functor, because it is not a comparison-based algorithm.
    * Only \ref std::less and \ref std::greater are accepted, they select
    * the ascending and the descending order respectively.
    *
    * \tparam Array is the type of container to be sorted (see \ref sort(Array&)).
    * \tparam Compare is the comparison functor, only `std::less` (or
    * `std::less<>`) and `std::greater` (or `std::greater<>`) are supported.
    * \param array is an instance of the container to be sorted in place.
    * \param compare is an instance of the comparison functor selecting the order.
    */
   template< typename Array, typename Compare >
   void static sort( Array& array, [[maybe_unused]] const Compare& compare )
   {
      static_assert(
         detail::is_std_less_v< Compare > || detail::is_std_greater_v< Compare >,
         "RadixSort supports only the std::less and std::greater comparison functors" );
      if constexpr( detail::is_std_greater_v< Compare > )
         detail::radixSort< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType, true >(
            array.getView() );
      else
         detail::radixSort< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType, false >(
            array.getView() );
   }

   /**
    * \brief Sorts `array` in ascending order, reusing the given workspace
    * of temporary device buffers.
    *
    * See the overload taking also the comparison functor for details.
    *
    * \tparam Array is the type of container to be sorted (see \ref sort(Array&)).
    * \param array is an instance of the container to be sorted in place.
    * \param workspace is the scratch space reusable across sorts; it must
    * have the same value, device and index types as `array`.
    */
   template< typename Array >
   void static sort(
      Array& array,
      Workspace< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType >& workspace )
   {
      sort( array, workspace, std::less{} );
   }

   /**
    * \brief Sorts `array` in ascending or descending order, reusing the
    * given workspace of temporary device buffers.
    *
    * Behaves like \ref sort(Array&, const Compare&), but takes the
    * temporary device buffers from \p workspace instead of allocating them
    * per call. The workspace is grown to fit `array` and keeps its
    * capacity for the next call, so a workspace reused across sorts of
    * arrays no larger than the largest one sorted so far performs no
    * device memory allocation at all.
    *
    * \tparam Array is the type of container to be sorted (see \ref sort(Array&)).
    * \tparam Compare is the comparison functor, only `std::less` (or
    * `std::less<>`) and `std::greater` (or `std::greater<>`) are supported.
    * \param array is an instance of the container to be sorted in place.
    * \param workspace is the scratch space reusable across sorts; it must
    * have the same value, device and index types as `array`.
    * \param compare is an instance of the comparison functor selecting the order.
    */
   template< typename Array, typename Compare >
   void static sort(
      Array& array,
      Workspace< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType >& workspace,
      [[maybe_unused]] const Compare& compare )
   {
      static_assert(
         detail::is_std_less_v< Compare > || detail::is_std_greater_v< Compare >,
         "RadixSort supports only the std::less and std::greater comparison functors" );
      if constexpr( detail::is_std_greater_v< Compare > )
         detail::radixSort< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType, true >(
            array.getView(), workspace );
      else
         detail::radixSort< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType, false >(
            array.getView(), workspace );
   }
};

}  // namespace TNL::Algorithms::Sorting
