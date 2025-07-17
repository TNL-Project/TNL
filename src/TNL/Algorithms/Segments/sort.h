// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Sort elements within specified segments in a range.
 *
 * This function sorts elements within segments in the range [\e begin, \e end). Each segment
 * is sorted independently using insertion sort. The sorting is performed based on the
 * provided \e fetch, \e compare, and \e swap functions.
 * The sorting is done in ascending order based on the comparison function provided.
 *
 * \tparam Segments Type of the segments container.
 * \tparam IndexBegin Type of the begin index.
 * \tparam IndexEnd Type of the end index.
 * \tparam Fetch Type of the fetch function.
 * \tparam Compare Type of the comparison function.
 * \tparam Swap Type of the swap function.
 *
 * \param segments The segments container.
 * \param begin The beginning of the range of segments to sort.
 * \param end The end of the range of segments to sort.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param compare Function to compare two elements. Should have signature:
 *        `bool compare(ValueType a, ValueType b)` returning true if a <= b.
 * \param swap Function to swap two elements. Should have signature:
 *        `void swap(IndexType globalIdx1, IndexType globalIdx2)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_sort.cpp
 *
 * \par Output
 * \include SegmentsExample_sort.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Compare,
          typename Swap,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
sortSegments( const Segments& segments,
              IndexBegin begin,
              IndexEnd end,
              Fetch&& fetch,
              Compare&& compare,
              Swap&& swap,
              LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Sort elements within all segments.
 *
 * This is a convenience function that sorts elements in all segments. It internally
 * calls \e sortSegments with the full range of segments.
 * The sorting is done in ascending order based on the comparison function provided.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Fetch Type of the fetch function.
 * \tparam Compare Type of the comparison function.
 * \tparam Swap Type of the swap function.
 *
 * \param segments The segments container.
 * \param fetch Function to fetch element value at given position.
 * \param compare Function to compare two elements.
 * \param swap Function to swap two elements.
 * \param launchConfig Configuration for parallel execution.
 *
 * See \ref sortSegments for a complete example.
 */
template< typename Segments, typename Fetch, typename Compare, typename Swap >
static void
sortAllSegments( const Segments& segments,
                 Fetch&& fetch,
                 Compare&& compare,
                 Swap&& swap,
                 LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Sort elements within specified segments using a segment index array.
 *
 * This function sorts elements within segments specified by the \e segmentIndexes array.
 * Each specified segment is sorted independently using insertion sort.
 * The sorting is done in ascending order based on the comparison function provided.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Array Type of the segment indexes array.
 * \tparam IndexBegin Type of the begin index.
 * \tparam IndexEnd Type of the end index.
 * \tparam Fetch Type of the fetch function.
 * \tparam Compare Type of the comparison function.
 * \tparam Swap Type of the swap function.
 *
 * \param segments The segments container.
 * \param segmentIndexes Array containing indices of segments to sort.
 * \param begin The beginning of the range in segmentIndexes.
 * \param end The end of the range in segmentIndexes.
 * \param fetch Function to fetch element value at given position.
 * \param compare Function to compare two elements.
 * \param swap Function to swap two elements.
 * \param launchConfig Configuration for parallel execution.
 *
 * See \ref sortSegments for a complete example.
 */
template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Compare,
          typename Swap,
          typename T = std::enable_if_t< IsArrayType< Array >::value
                                         && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
sortSegments( const Segments& segments,
              const Array& segmentIndexes,
              IndexBegin begin,
              IndexEnd end,
              Fetch&& fetch,
              Compare&& compare,
              Swap&& swap,
              LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Sort elements within all segments specified by a segment index array.
 *
 * This is a convenience function that sorts elements in all segments specified by
 * the \e segmentIndexes array. It internally calls \e sortSegments with the full range
 * of the \e segmentIndexes array.
 * The sorting is done in ascending order based on the comparison function provided.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Array Type of the segment indexes array.
 * \tparam Fetch Type of the fetch function.
 * \tparam Compare Type of the comparison function.
 * \tparam Swap Type of the swap function.
 *
 * \param segments The segments container.
 * \param segmentIndexes Array containing indices of segments to sort.
 * \param fetch Function to fetch element value at given position.
 * \param compare Function to compare two elements.
 * \param swap Function to swap two elements.
 * \param launchConfig Configuration for parallel execution.
 *
 * See \ref sortSegments for a complete example.
 */
template< typename Segments,
          typename Array,
          typename Fetch,
          typename Compare,
          typename Swap,
          typename T = std::enable_if_t< IsArrayType< Array >::value > >
static void
sortSegments( const Segments& segments,
              const Array& segmentIndexes,
              Fetch&& fetch,
              Compare&& compare,
              Swap&& swap,
              LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Sort elements within segments that satisfy a condition.
 *
 * This function sorts elements within segments in the range [\e begin, \e end) that satisfy
 * the given condition. Each qualifying segment is sorted independently using insertion sort.
 * The sorting is done in ascending order based on the comparison function provided.
 *
 * \tparam Segments Type of the segments container.
 * \tparam IndexBegin Type of the begin index.
 * \tparam IndexEnd Type of the end index.
 * \tparam Condition Type of the condition function.
 * \tparam Fetch Type of the fetch function.
 * \tparam Compare Type of the comparison function.
 * \tparam Swap Type of the swap function.
 *
 * \param segments The segments container.
 * \param begin The beginning of the range of segments to check.
 * \param end The end of the range of segments to check.
 * \param condition Function that determines if a segment should be sorted.
 *        Should have signature: `bool condition(IndexType segmentIdx)`.
 * \param fetch Function to fetch element value at given position.
 * \param compare Function to compare two elements.
 * \param swap Function to swap two elements.
 * \param launchConfig Configuration for parallel execution.
 *
 * See \ref sortSegments for a complete example.
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Compare,
          typename Swap,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
sortSegmentsIf( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Condition&& condition,
                Fetch&& fetch,
                Compare&& compare,
                Swap&& swap,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Sort elements within all segments that satisfy a condition.
 *
 * This is a convenience function that sorts elements in all segments that satisfy
 * the given condition. It internally calls \e sortSegmentsIf with the full range of segments.
 * The sorting is done in ascending order based on the comparison function provided.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Condition Type of the condition function.
 * \tparam Fetch Type of the fetch function.
 * \tparam Compare Type of the comparison function.
 * \tparam Swap Type of the swap function.
 *
 * \param segments The segments container.
 * \param condition Function that determines if a segment should be sorted.
 * \param fetch Function to fetch element value at given position.
 * \param compare Function to compare two elements.
 * \param swap Function to swap two elements.
 * \param launchConfig Configuration for parallel execution.
 *
 * See \ref sortSegments for a complete example.
 */
template< typename Segments, typename Condition, typename Fetch, typename Compare, typename Swap >
static void
sortAllSegmentsIf( const Segments& segments,
                   Condition&& condition,
                   Fetch&& fetch,
                   Compare&& compare,
                   Swap&& swap,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Sorts a segment using insertion sort.
 *
 * This function sorts the elements of a segment using insertion sort algorithm.
 *
 * \tparam SegmentView Type of the segment view.
 * \tparam Fetch Type of the fetch function.
 * \tparam Compare Type of the comparison function.
 * \tparam Swap Type of the swap function.
 *
 * \param segment The segment view to be sorted.
 * \param fetch Function to fetch element value at given position. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * \param compare Function to compare two elements.
 *        Should have signature: `bool compare(ValueType a, ValueType b)` returning true if a <= b.
 * \param swap Function to swap two elements.
 *        Should have signature: `void swap(IndexType globalIdx1, IndexType globalIdx2)`.
 *
 * This function performs an in-place sort of the segment using the insertion sort algorithm.
 * The sorting is done in ascending order based on the comparison function provided.
 */
template< typename SegmentView, typename Fetch, typename Compare, typename Swap >
__cuda_callable__
void
segmentInsertionSort( SegmentView segment, Fetch&& fetch, Compare&& compare, Swap&& swap );

}  // namespace TNL::Algorithms::Segments

#include "sort.hpp"
