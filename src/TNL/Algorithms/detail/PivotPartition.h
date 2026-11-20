// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* Host and GPU implementations of the three-way pivot partitioning.
 *
 * The functions below implement the public pivotPartition overloads (see
 * TNL/Algorithms/pivotPartition.h). The host implementations are
 * sequential - the classification counts and the scatter positions cannot
 * be computed by a single parallel pass over the data without auxiliary
 * storage, and the out-of-place host variant deliberately keeps the two
 * input passes of the counting/scatter scheme.
 *
 * The GPU implementation is grid-wide: the input is split into blocks of
 * a fixed number of elements, each block classifies its tile with the
 * cooperative block scan from PivotPartitionBlock.h, the per-block counts
 * are scanned exclusively over the whole grid, and a second pass scatters
 * each block into the output ranges computed from its scanned offset.
 *
 * The dispatchers at the bottom of this file route the calls according to
 * the device type of the arrays. The in-place GPU overload is
 * scratch-based: it partitions out of place and copies the result back,
 * since a direct allocation-free in-place partitioner is a research
 * project of its own.
 */

#include <algorithm>    // std::max
#include <cstddef>      // std::size_t
#include <stdexcept>    // std::invalid_argument
#include <type_traits>  // std::is_same_v, std::remove_const_t
#include <utility>      // std::pair, std::swap

#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/detail/PivotPartitionBlock.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Assert.h>
#include <TNL/Backend/DeviceInfo.h>
#include <TNL/Backend/KernelLaunch.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/DiscreteMath.h>  // TNL::roundUpDivision
#include <TNL/Math.h>          // TNL::min

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

#if defined( __CUDACC__ ) || defined( __HIP__ )

//! \brief Number of threads per block used by the grid-wide GPU partitioning.
inline constexpr int pivotPartitionThreadsPerBlock = 256;

//! \brief Number of input elements processed by each block
//! of the grid-wide GPU partitioning.
inline constexpr int pivotPartitionElementsPerBlock = 1024;

/**
 * \brief Counts the elements less, equivalent and greater than the pivot
 * in one tile of the grid-wide GPU partitioning.
 *
 * The input range of the block `blockIdx.x` is
 * `[ pivotPartitionElementsPerBlock * blockIdx.x, min( ( blockIdx.x + 1 ) * pivotPartitionElementsPerBlock, size ) )`.
 * All threads of the block cooperatively classify the tile, and thread 0
 * stores the totals of the block into `counts[ blockIdx.x ]`.
 *
 * \tparam Value The type of the partitioned elements.
 * \tparam Index The type used for indexing the partitioned arrays.
 * \tparam Count The type used for the classification counts.
 * \tparam Compare Type of the strict weak ordering comparator.
 *
 * \param input View of the whole input array.
 * \param counts View of the array of the per-block classification counts.
 * \param pivot The pivot value used for the classification.
 * \param compare Strict weak ordering used for the classification.
 */
template< typename Value, typename Index, typename Count, typename Compare >
__global__
void
pivotPartitionCountKernel(
   Containers::ArrayView< const Value, Devices::GPU, Index > input,
   Containers::ArrayView< PivotPartitionCounts< Count >, Devices::GPU, Index > counts,
   Value pivot,
   Compare compare )
{
   // the tile bounds are computed in a wider type to avoid overflow near the
   // limits of a narrow Index
   const std::size_t begin = static_cast< std::size_t >( pivotPartitionElementsPerBlock ) * blockIdx.x;
   const std::size_t end = TNL::min( begin + pivotPartitionElementsPerBlock, static_cast< std::size_t >( input.getSize() ) );

   const auto scan = scanPivotPartitionBlock< PivotPartitionEqualPolicy::Preserve, Count >(
      input.getView( static_cast< Index >( begin ), static_cast< Index >( end ) ), pivot, compare );

   if( threadIdx.x == 0 )
      counts[ blockIdx.x ] = scan.total;
}

/**
 * \brief Scatters one tile of the grid-wide GPU partitioning
 * into the output ranges of its buckets.
 *
 * The output bases of the block are computed from the exclusive scan of the
 * per-block counts: the lesser elements of all blocks precede the equivalent
 * elements, which precede the greater elements. The elements are re-classified
 * by \ref scanPivotPartitionBlock and scattered by
 * \ref scatterPivotPartitionBlock.
 *
 * \tparam Value The type of the partitioned elements.
 * \tparam Index The type used for indexing the partitioned arrays.
 * \tparam Count The type used for the classification counts.
 * \tparam Compare Type of the strict weak ordering comparator.
 * \tparam useShared Whether to stage the elements in the shared memory.
 *
 * \param input View of the whole input array.
 * \param output View of the whole output array.
 * \param offsets View of the exclusive scan of the per-block counts.
 * \param pivot The pivot value used for the classification.
 * \param compare Strict weak ordering used for the classification.
 * \param totals Grand totals of the classification counts.
 */
template< typename Value, typename Index, typename Count, typename Compare, bool useShared >
__global__
void
pivotPartitionScatterKernel(
   Containers::ArrayView< const Value, Devices::GPU, Index > input,
   Containers::ArrayView< Value, Devices::GPU, Index > output,
   Containers::ArrayView< PivotPartitionCounts< Count >, Devices::GPU, Index > offsets,
   Value pivot,
   Compare compare,
   PivotPartitionCounts< Count > totals )
{
   // named differently than the buffers in the other kernels, because all
   // extern __shared__ declarations of the same name in a translation unit
   // must be compatible, but this one is aligned for arbitrary Value types
   extern __shared__ __align__( 16 ) unsigned char stagingMem[];
   Value* sharedMemory = reinterpret_cast< Value* >( stagingMem );

   // the tile bounds are computed in a wider type to avoid overflow near the
   // limits of a narrow Index
   const std::size_t begin = static_cast< std::size_t >( pivotPartitionElementsPerBlock ) * blockIdx.x;
   const std::size_t end = TNL::min( begin + pivotPartitionElementsPerBlock, static_cast< std::size_t >( input.getSize() ) );

   // output bases of this block: the blocks scatter the lesser elements
   // first, then the equivalent ones, then the greater ones
   const PivotPartitionCounts< Count > blockOffset = offsets[ blockIdx.x ];
   const Index lessBase = static_cast< Index >( blockOffset.less );
   const Index equalBase = static_cast< Index >( totals.less + blockOffset.equal );
   const Index greaterBase = static_cast< Index >( totals.less + totals.equal + blockOffset.greater );

   const auto tile = input.getView( static_cast< Index >( begin ), static_cast< Index >( end ) );
   const auto scan = scanPivotPartitionBlock< PivotPartitionEqualPolicy::Preserve, Count >( tile, pivot, compare );
   scatterPivotPartitionBlock< PivotPartitionEqualPolicy::Preserve, useShared >(
      tile, output, pivot, compare, scan, lessBase, equalBase, greaterBase, sharedMemory );
}

/**
 * \brief Grid-wide out-of-place GPU implementation of the three-way pivot
 * partitioning.
 *
 * The input is processed in two kernel passes: the first one classifies the
 * elements of each block and stores the per-block counts, a device-wide
 * exclusive scan of the counts produces the output bases of each block, and
 * the second pass re-classifies the elements and scatters them into the
 * output ranges of their buckets. The elements are staged in the shared
 * memory only if a whole tile fits the shared-memory budget, otherwise they
 * are written to the output directly.
 *
 * The input and the output must not overlap. All kernels are launched
 * synchronously, so the output and the returned boundaries are complete
 * when the function returns.
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
pivotPartitionGpu(
   const InputArray& input,
   OutputArray& output,
   const typename InputArray::ValueType& pivot,
   const Compare& compare )
{
   using Index = typename OutputArray::IndexType;
   using Value = std::remove_const_t< typename InputArray::ValueType >;
   // the per-block counts must be at least int-sized: the device-wide scan
   // requires the value size to be a multiple of sizeof( int ), and count
   // arithmetic in a narrower type would be prone to overflow
   using Count = std::conditional_t< ( sizeof( Index ) >= sizeof( int ) ), Index, int >;
   using Counts = PivotPartitionCounts< Count >;

   static_assert(
      std::is_same_v< Index, typename InputArray::IndexType >,
      "pivotPartition: input and output must use the same index type" );

   const Index size = input.getSize();

   // zero-size input: no kernel is launched
   if( size == 0 )
      return { 0, 0 };

   // split the input into blocks of pivotPartitionElementsPerBlock elements
   const Index blocks = TNL::roundUpDivision( size, static_cast< Index >( pivotPartitionElementsPerBlock ) );

   // check the block count before converting it to the grid dimension,
   // the conversion of an out-of-range count would silently wrap around
   if( static_cast< std::size_t >( blocks ) > Backend::getMaxGridXSize( pivotPartitionThreadsPerBlock ) )
      throw std::invalid_argument( "pivotPartition: the input is too large for the GPU grid limits" );

   // classification counts of each block and the exclusive scan producing
   // the output bases of each block
   Containers::Array< Counts, Devices::GPU, Index > counts( blocks );
   Containers::Array< Counts, Devices::GPU, Index > offsets( blocks );

   const Containers::ArrayView< const Value, Devices::GPU, Index > inputView = input.getConstView();
   const Containers::ArrayView< Value, Devices::GPU, Index > outputView = output.getView();

   Backend::LaunchConfiguration launch_config;
   launch_config.blockSize.x = pivotPartitionThreadsPerBlock;
   launch_config.gridSize.x = static_cast< unsigned int >( blocks );

   // pass 1: classify the elements of each block
   constexpr auto countKernel = pivotPartitionCountKernel< Value, Index, Count, Compare >;
   Backend::launchKernelSync( countKernel, launch_config, inputView, counts.getView(), pivot, compare );

   // device-wide exclusive scan of the per-block counts: the block bases
   Algorithms::exclusiveScan( counts, offsets, 0, blocks, 0, PivotPartitionCountsPlus{}, Counts{ 0, 0, 0 } );

   // grand totals: the counts of the last block plus its scanned offset
   const Counts lastBlockCounts = counts.getElement( blocks - 1 );
   const Counts lastBlockOffset = offsets.getElement( blocks - 1 );
   const Counts totals = PivotPartitionCountsPlus{}( lastBlockCounts, lastBlockOffset );

   // pass 2: re-classify the elements and scatter them into the output
   // ranges of their buckets, staged in the shared memory only if a whole
   // tile fits the budget; the block scan allocates its small storage
   // statically, which is covered by a conservative reserve
   const int sharedReserve = static_cast< int >( sizeof( Count ) ) * ( 16 + 3 * Backend::getWarpSize() );
   const int sharedMemPerBlock = static_cast< int >( Backend::getSharedMemoryPerBlock( Backend::getDevice() ) );
   const int maxSharable = std::max( sharedMemPerBlock - sharedReserve, 0 );
   const std::size_t tileBytes = static_cast< std::size_t >( pivotPartitionElementsPerBlock ) * sizeof( Value );

   if( tileBytes <= static_cast< std::size_t >( maxSharable ) ) {
      launch_config.dynamicSharedMemorySize = tileBytes;
      constexpr auto scatterKernel = pivotPartitionScatterKernel< Value, Index, Count, Compare, true >;
      Backend::launchKernelSync(
         scatterKernel, launch_config, inputView, outputView, offsets.getView(), pivot, compare, totals );
   }
   else {
      launch_config.dynamicSharedMemorySize = 0;
      constexpr auto scatterKernel = pivotPartitionScatterKernel< Value, Index, Count, Compare, false >;
      Backend::launchKernelSync(
         scatterKernel, launch_config, inputView, outputView, offsets.getView(), pivot, compare, totals );
   }

   return { static_cast< Index >( totals.less ), static_cast< Index >( totals.less + totals.equal ) };
}

/**
 * \brief Scratch-based in-place GPU implementation of the three-way pivot
 * partitioning.
 *
 * A direct allocation-free in-place GPU partitioner would have to resolve
 * data-dependent permutation cycles cooperatively, so this overload simply
 * partitions into a scratch array of the same size and copies the result
 * back to the data.
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
pivotPartitionInPlaceGpu( Array& data, const typename Array::ValueType& pivot, const Compare& compare )
{
   using Index = typename Array::IndexType;
   using Value = std::remove_const_t< typename Array::ValueType >;
   using Device = std::remove_const_t< typename Array::DeviceType >;

   // the pivot may alias an element of the data, so it must be copied
   // before the elements start to move
   const Value pivotCopy = pivot;

   // partition out of place and copy the result back
   Containers::Array< Value, Device, Index > temporary( data.getSize() );
   const auto result = pivotPartitionGpu( data, temporary, pivotCopy, compare );
   Algorithms::copy< Device >( data.getData(), temporary.getData(), data.getSize() );
   return result;
}

#endif

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
   else if constexpr( std::is_same_v< Device, Devices::GPU > ) {
#if defined( __CUDACC__ ) || defined( __HIP__ )
      return pivotPartitionGpu( input, output, pivot, compare );
#else
      static_assert( dependentFalse< Device >, "pivotPartition on the GPU requires compiling the call with nvcc or hipcc" );
#endif
   }
   else {
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
   else if constexpr( std::is_same_v< Device, Devices::GPU > ) {
#if defined( __CUDACC__ ) || defined( __HIP__ )
      return pivotPartitionInPlaceGpu( data, pivot, compare );
#else
      static_assert( dependentFalse< Device >, "pivotPartition on the GPU requires compiling the call with nvcc or hipcc" );
#endif
   }
   else {
      static_assert( dependentFalse< Device >, "pivotPartition is not implemented for this device type" );
   }
}

}  // namespace TNL::Algorithms::detail
