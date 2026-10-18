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
 * The implementation requires an auxiliary array of the same size
 * (\f$N\f$) as the input array and an array of per-block histograms, which
 * is small (at most 32 blocks per multiprocessor times 256 counters).
 * The total memory overhead is approximately \f$N\f$.
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
};

}  // namespace TNL::Algorithms::Sorting
