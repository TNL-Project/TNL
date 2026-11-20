// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* Cooperative block-level three-way partitioning around a pivot value.
 *
 * The functions below classify the elements of one input tile as less,
 * equivalent or greater than the pivot, scan the per-thread classification
 * counts across the CUDA block, and scatter the elements into the three
 * output ranges. They are the shared building blocks of the GPU quicksort
 * paths (see Sorting/experimental/detail) and of the grid-wide pivot
 * partitioning.
 *
 * The device functions are *cooperative* - all threads of the block must
 * participate in every call, otherwise they deadlock. Even empty tiles must
 * be processed by the whole block, since the barriers inside are
 * unconditional.
 */

#include <cstdint>
#include <ostream>
#include <type_traits>

#include <TNL/Backend/Macros.h>
#include <TNL/Functional.h>
#include <TNL/Algorithms/detail/GPUScan/GPUBlockScan.h>

namespace TNL::Algorithms::detail {

/**
 * \brief Policy for handling the elements equivalent to the pivot.
 *
 * With `Preserve`, the pivot-equivalent elements are scattered like the
 * other categories, so every input element is copied to the output.
 *
 * With `FillLater`, the equivalent elements are not written at all and the
 * equal range of the output is left to the caller, which must fill it with
 * the pivot (or otherwise complete it) before the output is valid. This is
 * an internal optimization for sorters which treat comparison-equivalent
 * elements as interchangeable.
 */
enum class PivotPartitionEqualPolicy : std::uint8_t
{
   Preserve,
   FillLater,
};

/**
 * \brief Counts of elements less, equivalent and greater than the pivot.
 *
 * \tparam Count Type used for the counts.
 */
template< typename Count >
struct PivotPartitionCounts
{
   //! \brief Number of elements less than the pivot.
   Count less;
   //! \brief Number of elements equivalent to the pivot.
   Count equal;
   //! \brief Number of elements greater than the pivot.
   Count greater;

   //! \brief Component-wise comparison, needed e.g. by assertions in \ref TNL::Algorithms::exclusiveScan.
   [[nodiscard]] __cuda_callable__
   bool
   operator==( const PivotPartitionCounts& other ) const
   {
      return less == other.less && equal == other.equal && greater == other.greater;
   }
};

/**
 * \brief Streams the counts, e.g. into assertion diagnostics.
 *
 * \tparam Count Type used for the counts.
 * \param str Output stream.
 * \param counts The counts to be printed.
 *
 * \return The output stream.
 */
template< typename Count >
std::ostream&
operator<<( std::ostream& str, const PivotPartitionCounts< Count >& counts )
{
   return str << "{ less = " << counts.less << ", equal = " << counts.equal << ", greater = " << counts.greater << " }";
}

/**
 * \brief Results of the cooperative block scan over the classification counts.
 *
 * \tparam Count Type used for the counts.
 */
template< typename Count >
struct PivotPartitionBlockScan
{
   //! \brief Counts of the elements assigned to the calling thread.
   PivotPartitionCounts< Count > local;
   //! \brief Inclusive scan of the per-thread counts across the CUDA block.
   PivotPartitionCounts< Count > inclusive;
   //! \brief Totals of the whole block.
   PivotPartitionCounts< Count > total;
};

/**
 * \brief Function object adding \ref PivotPartitionCounts component-wise.
 *
 * Suitable as the reduction of \ref TNL::Algorithms::exclusiveScan over
 * arrays of `PivotPartitionCounts`.
 */
struct PivotPartitionCountsPlus
{
   /**
    * \brief Returns the component-wise sum of two count structures.
    *
    * \tparam Count Type used for the counts.
    * \param lhs First operand.
    * \param rhs Second operand.
    *
    * \return The component-wise sum of `lhs` and `rhs`.
    */
   template< typename Count >
   [[nodiscard]] __cuda_callable__
   PivotPartitionCounts< Count >
   operator()( const PivotPartitionCounts< Count >& lhs, const PivotPartitionCounts< Count >& rhs ) const
   {
      return { lhs.less + rhs.less, lhs.equal + rhs.equal, lhs.greater + rhs.greater };
   }
};

#if defined( __CUDACC__ ) || defined( __HIP__ )

/**
 * \brief Classifies and scans the elements of one input tile around the pivot.
 *
 * Each element of `inputTile` is classified by two comparisons: elements for
 * which `compare( value, pivot )` holds are counted as *less*, elements for
 * which `compare( pivot, value )` holds are counted as *greater*, and the
 * remaining elements as *equivalent*. The per-thread counts are scanned
 * inclusively across the block and the block totals are broadcast to all
 * threads in the returned structure.
 *
 * With `Preserve`, the equivalent elements are scanned as well. With
 * `FillLater`, the third scan is skipped and the total number of equivalent
 * elements is derived from the tile size.
 *
 * This is a *cooperative* function - all threads of the block must call it,
 * even when the tile is empty.
 *
 * \tparam policy Handling of the pivot-equivalent elements.
 * \tparam Count Type used for the counts and the block scan.
 * \param inputTile View of the input tile processed by this block.
 * \param pivot The pivot value used for the classification.
 * \param compare Strict weak ordering used for the classification.
 *
 * \return Local counts of the calling thread, inclusive scans of the counts
 *         over the block prefix of the calling thread, and the totals of
 *         the whole block.
 */
template< PivotPartitionEqualPolicy policy, typename Count, typename InputView, typename Compare >
__device__
PivotPartitionBlockScan< Count >
scanPivotPartitionBlock( InputView inputTile, const typename InputView::ValueType& pivot, const Compare& compare )
{
   using BlockScan = Algorithms::detail::GPUBlockScan< Algorithms::detail::ScanType::Inclusive, 0, TNL::Plus, Count >;

   __shared__ typename BlockScan::Storage storage;
   __shared__ PivotPartitionCounts< Count > blockTotals;

   PivotPartitionBlockScan< Count > scan{};

   // the counter is widened so that a narrow Index cannot wrap in i += blockDim.x
   using Index = typename InputView::IndexType;
   using WideIndex = std::common_type_t< Index, unsigned int >;
   for( WideIndex i = threadIdx.x; i < static_cast< WideIndex >( inputTile.getSize() ); i += blockDim.x ) {
      const typename InputView::ValueType& value = inputTile[ static_cast< Index >( i ) ];
      if( compare( value, pivot ) )
         ++scan.local.less;
      else if( compare( pivot, value ) )
         ++scan.local.greater;
      else
         ++scan.local.equal;
   }

   // the scan synchronizes internally, so one storage can be reused for all scans
   scan.inclusive.less = BlockScan::scan( TNL::Plus(), 0, scan.local.less, threadIdx.x, storage );
   scan.inclusive.greater = BlockScan::scan( TNL::Plus(), 0, scan.local.greater, threadIdx.x, storage );
   if constexpr( policy == PivotPartitionEqualPolicy::Preserve )
      scan.inclusive.equal = BlockScan::scan( TNL::Plus(), 0, scan.local.equal, threadIdx.x, storage );

   if( threadIdx.x == blockDim.x - 1 ) {  // the last thread holds the block totals
      blockTotals.less = scan.inclusive.less;
      blockTotals.greater = scan.inclusive.greater;
      if constexpr( policy == PivotPartitionEqualPolicy::Preserve )
         blockTotals.equal = scan.inclusive.equal;
      else
         // equals are not scattered, their count follows from the tile size
         blockTotals.equal = static_cast< Count >( inputTile.getSize() ) - blockTotals.less - blockTotals.greater;
   }
   __syncthreads();
   scan.total = blockTotals;
   return scan;
}

/**
 * \brief Scatters the elements of one input tile into the three output ranges.
 *
 * Every element of `inputTile` is classified again, using the same two
 * comparisons as \ref scanPivotPartitionBlock, and written into the output
 * range of its category. With `Preserve`, every input element is copied and
 * the equivalent elements end up in
 * `output[ equalOutputBegin, equalOutputBegin + scan.total.equal )`.
 * With `FillLater`, the equivalent elements are not written and
 * `equalOutputBegin` is ignored - completing the equal range of the output
 * is the caller's obligation.
 *
 * With `useShared == true`, the elements are first staged in `sharedMemory`,
 * grouped by category, and then flushed to the output by block-strided
 * loops, which keeps the global writes coalesced. The staging buffer must
 * have the capacity of the whole tile, laid out as `[less | equal |
 * greater]` with `Preserve`, or `[less | greater]` with `FillLater`. With
 * `useShared == false`, the elements are written to the output directly.
 *
 * This is a *cooperative* function - all threads of the block must call it,
 * even when the tile is empty. The input and the output must not overlap.
 *
 * \tparam policy Handling of the pivot-equivalent elements.
 * \tparam useShared Whether to stage the elements in shared memory.
 * \tparam InputView Type of the view of the input tile.
 * \tparam OutputView Type of the view of the output range.
 * \tparam Compare Type of the comparator.
 * \tparam Count Type used for the counts (same as in `scan`).
 * \param inputTile View of the input tile processed by this block.
 * \param output Output view, the bases below are relative to its beginning.
 * \param pivot The pivot value used for the classification.
 * \param compare Strict weak ordering used for the classification.
 * \param scan Block scan obtained from \ref scanPivotPartitionBlock for the
 *             same tile, pivot and compare.
 * \param lessOutputBegin Base index of the output range of the lesser elements.
 * \param equalOutputBegin Base index of the output range of the equivalent
 *                         elements (unused with `FillLater`).
 * \param greaterOutputBegin Base index of the output range of the greater elements.
 * \param sharedMemory Staging buffer of the block (unused with `useShared == false`).
 */
template<
   PivotPartitionEqualPolicy policy,
   bool useShared,
   typename InputView,
   typename OutputView,
   typename Compare,
   typename Count >
__device__
void
scatterPivotPartitionBlock(
   InputView inputTile,
   OutputView output,
   const typename InputView::ValueType& pivot,
   const Compare& compare,
   PivotPartitionBlockScan< Count > scan,
   typename OutputView::IndexType lessOutputBegin,
   [[maybe_unused]] typename OutputView::IndexType equalOutputBegin,
   typename OutputView::IndexType greaterOutputBegin,
   [[maybe_unused]] typename OutputView::ValueType* sharedMemory )
{
   using Index = typename OutputView::IndexType;
   using Value = typename OutputView::ValueType;
   // the counters are widened so that a narrow Index cannot wrap in i += blockDim.x
   using WideIndex = std::common_type_t< Index, unsigned int >;

   if constexpr( useShared ) {
      // stage the elements in shared memory, grouped by category:
      // [ less | equal | greater ] with Preserve, [ less | greater ] with FillLater
      Count greaterSharedBegin = scan.total.less;
      if constexpr( policy == PivotPartitionEqualPolicy::Preserve )
         greaterSharedBegin += scan.total.equal;

      // exclusive offsets of the calling thread within the staged categories
      Count lessPosition = scan.inclusive.less - scan.local.less;
      [[maybe_unused]] Count equalPosition = 0;
      if constexpr( policy == PivotPartitionEqualPolicy::Preserve )
         equalPosition = scan.total.less + scan.inclusive.equal - scan.local.equal;
      Count greaterPosition = greaterSharedBegin + scan.inclusive.greater - scan.local.greater;

      for( WideIndex i = threadIdx.x; i < static_cast< WideIndex >( inputTile.getSize() ); i += blockDim.x ) {
         const Value& value = inputTile[ static_cast< Index >( i ) ];
         if( compare( value, pivot ) )
            sharedMemory[ lessPosition++ ] = value;
         else if( compare( pivot, value ) )
            sharedMemory[ greaterPosition++ ] = value;
         else if constexpr( policy == PivotPartitionEqualPolicy::Preserve )
            sharedMemory[ equalPosition++ ] = value;
      }
      __syncthreads();

      // flush the staged categories into their output ranges
      for( Count i = threadIdx.x; i < scan.total.less; i += blockDim.x )
         output[ lessOutputBegin + i ] = sharedMemory[ i ];
      if constexpr( policy == PivotPartitionEqualPolicy::Preserve ) {
         for( Count i = threadIdx.x; i < scan.total.equal; i += blockDim.x )
            output[ equalOutputBegin + i ] = sharedMemory[ scan.total.less + i ];
      }
      for( Count i = threadIdx.x; i < scan.total.greater; i += blockDim.x )
         output[ greaterOutputBegin + i ] = sharedMemory[ greaterSharedBegin + i ];
   }
   else {
      // exclusive offsets of the calling thread within the output ranges
      Index lessPosition = lessOutputBegin + static_cast< Index >( scan.inclusive.less - scan.local.less );
      [[maybe_unused]] Index equalPosition = equalOutputBegin;
      if constexpr( policy == PivotPartitionEqualPolicy::Preserve )
         equalPosition += static_cast< Index >( scan.inclusive.equal - scan.local.equal );
      Index greaterPosition = greaterOutputBegin + static_cast< Index >( scan.inclusive.greater - scan.local.greater );

      for( WideIndex i = threadIdx.x; i < static_cast< WideIndex >( inputTile.getSize() ); i += blockDim.x ) {
         const Value& value = inputTile[ static_cast< Index >( i ) ];
         if( compare( value, pivot ) )
            output[ lessPosition++ ] = value;
         else if( compare( pivot, value ) )
            output[ greaterPosition++ ] = value;
         else if constexpr( policy == PivotPartitionEqualPolicy::Preserve )
            output[ equalPosition++ ] = value;
      }
   }
}

#endif

}  // namespace TNL::Algorithms::detail
