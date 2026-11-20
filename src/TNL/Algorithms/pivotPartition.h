// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>   // std::less
#include <stdexcept>    // std::invalid_argument
#include <type_traits>  // std::enable_if_t, std::is_const_v, std::is_same_v, std::remove_const_t

#include <TNL/Algorithms/detail/PivotPartition.h>
#include <TNL/TypeTraits.h>  // IsArrayType

namespace TNL::Algorithms {

/**
 * \brief Result of the three-way partitioning around a pivot.
 *
 * The boundaries split the partitioned range into three consecutive buckets:
 * elements ordered before the pivot, elements equivalent to the pivot, and
 * elements ordered after the pivot. Both fields are boundaries relative to
 * the beginning of the output range, not bucket sizes.
 *
 * The aggregate supports structured bindings:
 *
 * \code{.cpp}
 * auto [ lessEnd, greaterBegin ] = pivotPartition( data, pivot );
 * \endcode
 *
 * \tparam Index Type used for indexing the partitioned data.
 */
template< typename Index >
struct PivotPartitionResult
{
   //! \brief End of the range of the elements ordered before the pivot.
   Index lessEnd;
   //! \brief Begin of the range of the elements ordered after the pivot.
   Index greaterBegin;
};

/**
 * \brief Partitions elements of an array into values less than, equivalent
 * to, and greater than a pivot, in place.
 *
 * Equivalence means that neither `compare( value, pivot )` nor
 * `compare( pivot, value )` is true, i.e. equality is not tested with
 * `operator==`.
 *
 * Every input element is preserved exactly once, only its position changes.
 * In particular, pivot-equivalent elements are moved rather than
 * reconstructed from the pivot, so comparison-equivalent elements may carry
 * distinct payloads.
 *
 * The operation is not stable; the order of the elements within each bucket
 * is unspecified.
 *
 * The returned boundaries are relative to the beginning of `data`:
 *
 * - `data[ 0, result.lessEnd )` contains the elements for which
 *   `compare( value, pivot )` is true;
 * - `data[ result.lessEnd, result.greaterBegin )` contains the elements
 *   equivalent to the pivot;
 * - `data[ result.greaterBegin, data.getSize() )` contains the elements for
 *   which `compare( pivot, value )` is true.
 *
 * The boundaries always satisfy
 * `0 <= result.lessEnd <= result.greaterBegin <= data.getSize()`. An empty
 * array returns `{ 0, 0 }`; a single element yields `{ 1, 1 }`, `{ 0, 1 }`
 * or `{ 0, 0 }` depending on whether it is ordered before, equivalent to,
 * or ordered after the pivot.
 *
 * The GPU implementation throws \ref std::invalid_argument when the input
 * is too large for the device grid limits.
 *
 * The pivot value is copied before the partitioning starts, so `pivot` may
 * refer to an element of `data`.
 *
 * `compare` must impose a strict weak ordering on the values.
 *
 * The operation completes synchronously - when the function returns, `data`
 * contains the partitioned data.
 *
 * Complexity: `O( N )` time and `O( 1 )` auxiliary storage, where `N` is the
 * size of `data`.
 *
 * \tparam Array is a type of container to be partitioned. It can be, for
 *         example, \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *         \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView.
 * \tparam Compare is a functor for comparing of two elements. It returns
 *         true if the first argument should be ordered before the second.
 *
 * \param data is an instance of array/array view/vector/vector view for
 *             partitioning.
 * \param pivot is the pivot value used for the classification of the
 *              elements.
 * \param compare is an instance of the functor for comparison of two
 *                elements.
 *
 * \return Boundaries of the less and greater buckets, see
 *         \ref PivotPartitionResult.
 */
template< typename Array, typename Compare = std::less<>, typename = std::enable_if_t< IsArrayType< Array >::value > >
[[nodiscard]] PivotPartitionResult< typename Array::IndexType >
pivotPartition( Array& data, const typename Array::ValueType& pivot, const Compare& compare = Compare{} )
{
   static_assert( ! std::is_const_v< Array >, "pivotPartition: the array to be partitioned in place must be non-const" );
   static_assert(
      ! std::is_const_v< typename Array::ValueType >,
      "pivotPartition: the array to be partitioned in place must have non-const elements" );
   auto [ lessEnd, greaterBegin ] = detail::pivotPartition( data, pivot, compare );
   return { lessEnd, greaterBegin };
}

/**
 * \brief Partitions elements of an array into values less than, equivalent
 * to, and greater than a pivot, into a separate output.
 *
 * Equivalence means that neither `compare( value, pivot )` nor
 * `compare( pivot, value )` is true, i.e. equality is not tested with
 * `operator==`.
 *
 * Every input element is copied to the output exactly once. In particular,
 * pivot-equivalent elements are copied rather than reconstructed from the
 * pivot, so comparison-equivalent elements may carry distinct payloads.
 * The input is left unchanged.
 *
 * The operation is not stable; the order of the elements within each bucket
 * is unspecified.
 *
 * The returned boundaries are relative to the beginning of `output`:
 *
 * - `output[ 0, result.lessEnd )` contains the elements for which
 *   `compare( value, pivot )` is true;
 * - `output[ result.lessEnd, result.greaterBegin )` contains the elements
 *   equivalent to the pivot;
 * - `output[ result.greaterBegin, output.getSize() )` contains the elements
 *   for which `compare( pivot, value )` is true.
 *
 * The boundaries always satisfy
 * `0 <= result.lessEnd <= result.greaterBegin <= output.getSize()`. An empty
 * input returns `{ 0, 0 }`; a single element yields `{ 1, 1 }`, `{ 0, 1 }`
 * or `{ 0, 0 }` depending on whether it is ordered before, equivalent to,
 * or ordered after the pivot.
 *
 * The GPU implementation throws \ref std::invalid_argument when the input
 * is too large for the device grid limits.
 *
 * The input and output ranges must not overlap; use the in-place overload
 * to partition the data in place. The arrays must be allocated on the same
 * device, use the same index type and have the same value type, and the
 * output elements must be non-const. If the sizes of `input` and `output`
 * differ, \ref std::invalid_argument is thrown.
 *
 * `compare` must impose a strict weak ordering on the values.
 *
 * The operation completes synchronously - when the function returns,
 * `output` contains the partitioned data.
 *
 * Complexity: `O( N )` time, two passes over the input and `O( 1 )`
 * auxiliary storage, where `N` is the size of `input`.
 *
 * \tparam InputArray is a type of container with the input elements. It can
 *         be, for example, \ref TNL::Containers::Array,
 *         \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector,
 *         or \ref TNL::Containers::VectorView.
 * \tparam OutputArray is a type of container for the output elements, with
 *         the same restrictions as InputArray.
 * \tparam Compare is a functor for comparing of two elements. It returns
 *         true if the first argument should be ordered before the second.
 *
 * \param input is an instance of array/array view/vector/vector view with
 *              the elements to be partitioned.
 * \param output is an instance of array/array view/vector/vector view for
 *               the partitioned elements.
 * \param pivot is the pivot value used for the classification of the
 *              elements.
 * \param compare is an instance of the functor for comparison of two
 *                elements.
 *
 * \return Boundaries of the less and greater buckets, see
 *         \ref PivotPartitionResult.
 */
template<
   typename InputArray,
   typename OutputArray,
   typename Compare = std::less<>,
   typename = std::enable_if_t< IsArrayType< InputArray >::value && IsArrayType< OutputArray >::value > >
[[nodiscard]] PivotPartitionResult< typename OutputArray::IndexType >
pivotPartition(
   const InputArray& input,
   OutputArray& output,
   const typename InputArray::ValueType& pivot,
   const Compare& compare = Compare{} )
{
   static_assert(
      std::is_same_v< typename InputArray::DeviceType, typename OutputArray::DeviceType >,
      "pivotPartition: input and output must be allocated on the same device" );
   static_assert(
      std::is_same_v< typename InputArray::IndexType, typename OutputArray::IndexType >,
      "pivotPartition: input and output must use the same index type" );
   static_assert(
      std::is_same_v<
         std::remove_const_t< typename InputArray::ValueType >,
         std::remove_const_t< typename OutputArray::ValueType > >,
      "pivotPartition: input and output must have the same value type" );
   static_assert(
      ! std::is_const_v< typename OutputArray::ValueType >, "pivotPartition: the output must have non-const elements" );
   static_assert( ! std::is_const_v< OutputArray >, "pivotPartition: the output must be non-const" );

   if( input.getSize() != output.getSize() )
      throw std::invalid_argument( "pivotPartition: input and output must have the same size" );

   auto [ lessEnd, greaterBegin ] = detail::pivotPartition( input, output, pivot, compare );
   return { lessEnd, greaterBegin };
}

}  // namespace TNL::Algorithms
