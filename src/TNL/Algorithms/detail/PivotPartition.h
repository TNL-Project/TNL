// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* Host implementations of the three-way pivot partitioning.
 *
 * The functions below implement the public pivotPartition overloads (see
 * TNL/Algorithms/pivotPartition.h) for the host execution models. They are
 * sequential - the classification counts and the scatter positions cannot
 * be computed by a single parallel pass over the data without auxiliary
 * storage, and the out-of-place host variant deliberately keeps the two
 * input passes of the counting/scatter scheme.
 *
 * The dispatchers at the bottom of this file route the calls according to
 * the device type of the arrays. The grid-wide GPU implementation lands in
 * the next commit, so the GPU branch only provides a diagnostic
 * static_assert.
 */

#include <type_traits>  // std::is_same_v, std::remove_const_t
#include <utility>      // std::pair, std::swap

#include <TNL/Assert.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>

namespace TNL::Algorithms::detail {

//! \brief Dependent false value for static assertions in dispatch branches
//! which must fail only for unsupported device types.
template< typename >
inline constexpr bool dependentFalse = false;

/**
 * \brief Out-of-place host implementation of the three-way pivot partitioning.
 *
 * The input is traversed twice: the first pass counts the elements which are
 * less than, equivalent to, and greater than the pivot, the second pass
 * scatters each element into the output range of its bucket. Every element
 * is written exactly once, so the output is a permutation of the input and
 * no element is reconstructed from the pivot.
 *
 * The input and the output must not overlap.
 *
 * \tparam InputArray Type of the input array or view.
 * \tparam OutputArray Type of the output array or view.
 * \tparam Compare Type of the strict weak ordering comparator.
 *
 * \param input Array or view with the elements to be partitioned.
 * \param output Output array or view, must have the same size as `input`.
 * \param pivot The pivot value used for the classification.
 * \param compare Strict weak ordering used for the classification.
 *
 * \return Pair of boundaries `{ lessEnd, greaterBegin }` relative to the
 *         beginning of the output range.
 */
template< typename InputArray, typename OutputArray, typename Compare >
std::pair< typename OutputArray::IndexType, typename OutputArray::IndexType >
pivotPartitionHost(
   const InputArray& input,
   OutputArray& output,
   const typename InputArray::ValueType& pivot,
   const Compare& compare )
{
   using Index = typename OutputArray::IndexType;
   using Value = std::remove_const_t< typename InputArray::ValueType >;

   const Value* inputData = input.getData();
   Value* outputData = output.getData();
   const Index size = input.getSize();

   // counting pass: classify each element of the input
   Index lessTotal = 0;
   Index equalTotal = 0;
   // only used by the assertion below, which is compiled out with NDEBUG
   [[maybe_unused]] Index greaterTotal = 0;
   for( Index i = 0; i < size; i++ ) {
      if( compare( inputData[ i ], pivot ) )
         ++lessTotal;
      else if( compare( pivot, inputData[ i ] ) )
         ++greaterTotal;
      else
         ++equalTotal;
   }
   TNL_ASSERT_EQ( lessTotal + equalTotal + greaterTotal, size, "the buckets must cover the whole input" );

   // scatter pass: write each element exactly once into its bucket,
   // the equal bucket starts right after the less bucket
   Index lessPosition = 0;
   Index equalPosition = lessTotal;
   Index greaterPosition = lessTotal + equalTotal;
   for( Index i = 0; i < size; i++ ) {
      const Value& value = inputData[ i ];
      if( compare( value, pivot ) )
         outputData[ lessPosition++ ] = value;
      else if( compare( pivot, value ) )
         outputData[ greaterPosition++ ] = value;
      else
         outputData[ equalPosition++ ] = value;
   }
   TNL_ASSERT_EQ( lessPosition, lessTotal, "the scatter pass wrote a wrong number of lesser elements" );
   TNL_ASSERT_EQ( equalPosition, lessTotal + equalTotal, "the scatter pass wrote a wrong number of equivalent elements" );
   TNL_ASSERT_EQ( greaterPosition, size, "the scatter pass wrote a wrong number of greater elements" );

   return { lessTotal, lessTotal + equalTotal };
}

/**
 * \brief In-place host implementation of the three-way pivot partitioning.
 *
 * The elements are partitioned by the Dutch national flag algorithm: the
 * range `[0, less)` holds the lesser elements, `[less, current)` the
 * equivalent ones, `[current, greater)` the elements which were not
 * classified yet, and `[greater, size)` the greater ones. The unclassified
 * range shrinks with every step until it is empty.
 *
 * \tparam Array Type of the array or view to be partitioned.
 * \tparam Compare Type of the strict weak ordering comparator.
 *
 * \param data Array or view with the elements to be partitioned.
 * \param pivot The pivot value used for the classification. It may alias an
 *              element of `data`, so it is copied before the partitioning
 *              starts.
 * \param compare Strict weak ordering used for the classification.
 *
 * \return Pair of boundaries `{ lessEnd, greaterBegin }` relative to the
 *         beginning of `data`.
 */
template< typename Array, typename Compare >
std::pair< typename Array::IndexType, typename Array::IndexType >
pivotPartitionInPlaceHost( Array& data, const typename Array::ValueType& pivot, const Compare& compare )
{
   using Index = typename Array::IndexType;
   using Value = std::remove_const_t< typename Array::ValueType >;

   Value* dataPtr = data.getData();
   const Index size = data.getSize();

   // the pivot may alias an element of the data, so it must be copied
   // before the elements start to move
   const Value pivotCopy = pivot;

   Index less = 0;
   Index current = 0;
   Index greater = size;
   while( current < greater ) {
      if( compare( dataPtr[ current ], pivotCopy ) )
         std::swap( dataPtr[ less++ ], dataPtr[ current++ ] );
      else if( compare( pivotCopy, dataPtr[ current ] ) )
         std::swap( dataPtr[ current ], dataPtr[ --greater ] );
      else
         ++current;
   }
   TNL_ASSERT_LE( less, current, "the less bucket must not cross the equal bucket" );
   TNL_ASSERT_EQ( current, greater, "all elements must be classified" );

   return { less, greater };
}

/**
 * \brief Dispatches the out-of-place three-way pivot partitioning
 * to the implementation of the arrays' device.
 *
 * \tparam InputArray Type of the input array or view.
 * \tparam OutputArray Type of the output array or view.
 * \tparam Compare Type of the strict weak ordering comparator.
 *
 * \param input Array or view with the elements to be partitioned.
 * \param output Output array or view, must have the same size as `input`.
 * \param pivot The pivot value used for the classification.
 * \param compare Strict weak ordering used for the classification.
 *
 * \return Pair of boundaries `{ lessEnd, greaterBegin }` relative to the
 *         beginning of the output range.
 */
template< typename InputArray, typename OutputArray, typename Compare >
std::pair< typename OutputArray::IndexType, typename OutputArray::IndexType >
pivotPartition(
   const InputArray& input,
   OutputArray& output,
   const typename InputArray::ValueType& pivot,
   const Compare& compare )
{
   using Device = typename OutputArray::DeviceType;
   if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > ) {
      return pivotPartitionHost( input, output, pivot, compare );
   }
   else {
      // the grid-wide GPU implementation lands in the next commit
      static_assert( dependentFalse< Device >, "pivotPartition is not implemented for this device type" );
   }
}

/**
 * \brief Dispatches the in-place three-way pivot partitioning
 * to the implementation of the array's device.
 *
 * \tparam Array Type of the array or view to be partitioned.
 * \tparam Compare Type of the strict weak ordering comparator.
 *
 * \param data Array or view with the elements to be partitioned.
 * \param pivot The pivot value used for the classification. It may alias an
 *              element of `data`.
 * \param compare Strict weak ordering used for the classification.
 *
 * \return Pair of boundaries `{ lessEnd, greaterBegin }` relative to the
 *         beginning of `data`.
 */
template< typename Array, typename Compare >
std::pair< typename Array::IndexType, typename Array::IndexType >
pivotPartition( Array& data, const typename Array::ValueType& pivot, const Compare& compare )
{
   using Device = typename Array::DeviceType;
   if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > ) {
      return pivotPartitionInPlaceHost( data, pivot, compare );
   }
   else {
      // the grid-wide GPU implementation lands in the next commit
      static_assert( dependentFalse< Device >, "pivotPartition is not implemented for this device type" );
   }
}

}  // namespace TNL::Algorithms::detail
