// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"

/**
 * \page SegmentSortOverview Overview of Segment Sort Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all sort functions available for segment operations,
 * helping to understand the differences between variants and choose the right function for your needs.
 *
 * \section SegmentSortFunctionCategories Function Categories
 *
 * The segment sort functions are organized into the following categories:
 *
 * \subsection SegmentSortBasicFunctions Basic Sort Functions
 *
 * These functions sort elements within segments:
 *
 * | Function | Segments Sorted | Description |
 * |----------|----------------|-------------|
 * | \ref sortAllSegments | All segments | Sorts all segments in the container |
 * | \ref sortSegments (range) | Segments [begin, end) | Sorts segments in a specified range |
 * | \ref sortSegments (array) | Segments in array | Sorts only segments whose indices are in the provided array |
 *
 * **When to use:**
 * - Use `sortAllSegments` when you need to sort all segments
 * - Use `sortSegments` with range when you want to sort a contiguous range of segments
 * - Use `sortSegments` with array when you have a specific, non-contiguous set of segment indices to sort
 *
 * \subsection SegmentSortConditionalFunctions Conditional Sort Functions
 *
 * These functions add a segment-level condition, sorting only segments that satisfy the condition:
 *
 * | Function | Segments Sorted | Description |
 * |----------|----------------|-------------|
 * | \ref sortAllSegmentsIf | All segments matching condition | Sorts all segments that satisfy the segment condition |
 * | \ref sortSegmentsIf | Segments [begin, end) matching condition | Sorts segments in range that satisfy the segment condition
 * |
 *
 * **When to use:**
 * - Use these variants when you want to skip sorting certain segments based on segment-level properties
 *
 * \subsection SegmentSortLowLevelFunctions Low-Level Functions
 *
 * | Function | Purpose |
 * |----------|---------|
 * | \ref segmentInsertionSort | Sorts a single segment view using insertion sort |
 *
 * **When to use:**
 * - Use this when you have a `SegmentView` object and need fine-grained control
 * - Useful when implementing custom segment algorithms
 *
 * The sorting order is determined by the comparison function (see \ref SegmentSortCompareLambda).
 *
 * \section SegmentSortParameters Common Parameters
 *
 * All sort functions share these common parameters:
 *
 * - **segments**: The segments container to sort
 * - **fetch**: Lambda that retrieves element values for comparison (see \ref SegmentSortFetchLambda)
 * - **compare**: Lambda that determines element ordering (see \ref SegmentSortCompareLambda)
 * - **swap**: Lambda that exchanges two elements (see \ref SegmentSortSwapLambda)
 * - **launchConfig**: Configuration for parallel execution (optional)
 *
 * Conditional variants additionally require:
 * - **condition**: Lambda that determines if a segment should be sorted (see \ref SegmentSortConditionLambda)
 *
 * \section SegmentSortRelatedPages Related Pages
 *
 * - \ref SegmentSortLambdas - Detailed lambda function signatures
 */

/**
 * \page SegmentSortLambdas Lambda Functions for Segment Sort Operations
 *
 * \tableofcontents
 *
 * This page describes the lambda function signatures used in segment sort operations.
 *
 * \section SegmentSortFetchLambda Fetch Lambda
 *
 * The fetch lambda retrieves the value of an element for comparison during sorting.
 * It has one of the following forms:
 *
 * **Full form:**
 * ```cpp
 * auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx )
 * {
 *    // Return the value at the given position
 *    return ...;
 * };
 * ```
 *
 * **Brief form:**
 * ```cpp
 * auto fetch = [=] __cuda_callable__ ( IndexType globalIdx )
 * {
 *    // Return the value at the given position
 *    return ...;
 * };
 * ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * \section SegmentSortCompareLambda Compare Lambda
 *
 * The compare lambda determines the ordering of two elements during sorting.
 * It has the following signature:
 *
 * ```cpp
 * auto compare = [=] __cuda_callable__ ( ValueType a, ValueType b ) -> bool
 * {
 *    // Return true if a should come before or equal to b
 *    return a <= b;
 * };
 * ```
 *
 * The lambda should return `true` if `a` should come before or be equal to `b` in the sorted order.
 *
 * \section SegmentSortSwapLambda Swap Lambda
 *
 * The swap lambda exchanges two elements during the sorting process.
 * It has the following signature:
 *
 * ```cpp
 * auto swap = [=] __cuda_callable__ ( IndexType globalIdx1, IndexType globalIdx2 )
 * {
 *    // Swap elements at positions globalIdx1 and globalIdx2
 *    ...
 * };
 * ```
 *
 * - \e globalIdx1 is the index of the first element to swap.
 * - \e globalIdx2 is the index of the second element to swap.
 *
 * \section SegmentSortConditionLambda Condition Lambda
 *
 * The condition lambda determines which segments should be sorted.
 * It has the following signature:
 *
 * ```cpp
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool
 * {
 *    // Return true if segment should be sorted
 *    return ...;
 * };
 * ```
 *
 * - \e segmentIdx is the index of the segment.
 *
 * The lambda should return `true` if the segment should be sorted.
 */

namespace TNL::Algorithms::Segments {
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
 * \param fetch Function to fetch element value at given position. See \ref SegmentSortFetchLambda.
 * \param compare Function to compare two elements. See \ref SegmentSortCompareLambda.
 * \param swap Function to swap two elements. See \ref SegmentSortSwapLambda.
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
 * \param fetch Function to fetch element value at given position. See \ref SegmentSortFetchLambda.
 * \param compare Function to compare two elements. See \ref SegmentSortCompareLambda.
 * \param swap Function to swap two elements. See \ref SegmentSortSwapLambda.
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
 * \param condition Function that determines if a segment should be sorted. See \ref SegmentSortConditionLambda.
 * \param fetch Function to fetch element value at given position. See \ref SegmentSortFetchLambda.
 * \param compare Function to compare two elements. See \ref SegmentSortCompareLambda.
 * \param swap Function to swap two elements. See \ref SegmentSortSwapLambda.
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
 * \param condition Function that determines if a segment should be sorted. See \ref SegmentSortConditionLambda.
 * \param fetch Function to fetch element value at given position. See \ref SegmentSortFetchLambda.
 * \param compare Function to compare two elements. See \ref SegmentSortCompareLambda.
 * \param swap Function to swap two elements. See \ref SegmentSortSwapLambda.
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
 * \param fetch Function to fetch element value at given position. See \ref SegmentSortFetchLambda.
 * \param compare Function to compare two elements. See \ref SegmentSortCompareLambda.
 * \param swap Function to swap two elements. See \ref SegmentSortSwapLambda.
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
