// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Sorting/detail/bitonicSort.h>

namespace TNL::Algorithms::Sorting {

/**
 * \brief Parallel [bitonic sort](https://en.wikipedia.org/wiki/Bitonic_sorter) on GPU.
 *
 * Bitonic sort is a sorting network:
 * the sequence of compared pairs does not depend on the data,
 * only the decision whether to swap a pair does.
 * It performs \f$ O( n \log^2 n ) \f$ comparisons,
 * which is more than the comparison-optimal algorithms,
 * but it needs no auxiliary memory and it sorts arrays of any size,
 * not only powers of two.
 * The sort is not stable.
 *
 * The class provides two interfaces:
 *
 * - \ref sort sorts a whole array on GPU.
 *   It runs one kernel per merge phase
 *   and uses the shared memory for all phases that fit into it.
 * - \ref inplaceSort sorts a range of indices
 *   accessed only through user-defined comparison and swap functions,
 *   which allows sorting of data that are not stored in one array,
 *   e.g. keys with a payload.
 */
struct BitonicSort
{
   /**
    * \brief Sorts an array in ascending order.
    *
    * \tparam Array Type of the array to be sorted.
    *               It must be \ref TNL::Containers::Array,
    *               \ref TNL::Containers::ArrayView or a similar container
    *               with the `TNL::Devices::GPU` device.
    *
    * \param array The array to be sorted in place.
    */
   template< typename Array >
   void static sort( Array& array )
   {
      sort( array, std::less<>{} );
   }

   /**
    * \brief Sorts an array using a custom comparison function.
    *
    * The array is sorted so that `compare( a, b )` is `true`
    * when `a` is placed before `b`,
    * i.e. `std::less<>` sorts in ascending order
    * and `std::greater<>` sorts in descending order.
    *
    * \tparam Array Type of the array to be sorted.
    *               It must be \ref TNL::Containers::Array,
    *               \ref TNL::Containers::ArrayView or a similar container
    *               with the `TNL::Devices::GPU` device.
    * \tparam Compare Type of the comparison function.
    *
    * \param array The array to be sorted in place.
    * \param compare Strict weak ordering of the array elements.
    *                It is called as `compare( a, b )` in device code,
    *                so it must be `__cuda_callable__`.
    *
    * \throws std::length_error if the array size rounded up to a power of two
    *         cannot be represented by the index type of the array.
    */
   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      detail::bitonicSort( array.getView(), compare );
   }

   /**
    * \brief Sorts a range of indices using custom comparison and swap functions.
    *
    * The sort does not access any data directly,
    * it only calls `compare( i, j )` and `swap( i, j )`
    * with indices `i` and `j` from the range `[begin, end)`.
    * The range is sorted so that `compare( i, j )` is `true`
    * when the element with index `i` is placed before the element with index `j`.
    * For example, keys with a payload can be sorted as follows:
    *
    * \code{.cpp}
    * auto compare = [ = ] __cuda_callable__( int i, int j ) { return keys[ i ] < keys[ j ]; };
    * auto swap = [ = ] __cuda_callable__( int i, int j ) mutable {
    *    TNL::swap( keys[ i ], keys[ j ] );
    *    TNL::swap( payload[ i ], payload[ j ] );
    * };
    * BitonicSort::inplaceSort< Devices::GPU >( 0, keys.getSize(), compare, swap );
    * \endcode
    *
    * This variant uses only the global memory,
    * so it is slower than \ref sort.
    *
    * \tparam Device Device where the data are stored.
    *                Only `TNL::Devices::GPU` is supported.
    * \tparam Index Type of the indices.
    * \tparam Compare Type of the comparison function.
    * \tparam Swap Type of the swap function.
    *
    * \param begin The first index of the range to be sorted.
    * \param end The index after the last index of the range to be sorted.
    * \param compare `__cuda_callable__` function which returns `true`
    *                when the element with the first index is placed
    *                before the element with the second index.
    * \param swap `__cuda_callable__` function which swaps the elements
    *             with the given indices.
    *
    * \throws std::length_error if the range size rounded up to a power of two
    *         cannot be represented by the index type used in the kernels.
    */
   template< typename Device, typename Index, typename Compare, typename Swap >
   void static inplaceSort( const Index begin, const Index end, const Compare& compare, const Swap& swap )
   {
      static_assert( std::is_same_v< Device, Devices::GPU >, "inplace bitonic sort is implemented only for GPU" );
      detail::bitonicSort( begin, end, compare, swap );
   }
};

}  // namespace TNL::Algorithms::Sorting
