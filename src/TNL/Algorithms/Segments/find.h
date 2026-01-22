// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"
#include "detail/FetchLambdaAdapter.h"

/**
 * \page SegmentFindOverview Overview of Segment Find Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all find functions available for segment operations,
 * helping to understand the differences between variants and choose the right function for your needs.
 *
 * \section SegmentFindFunctionCategories Function Categories
 *
 * The segment find functions are organized into the following categories:
 *
 * \subsection SegmentFindBasicFunctions Basic Find Functions
 *
 * These functions search for elements satisfying a condition within segments:
 *
 * | Function | Segments Searched | Description |
 * |----------|------------------|-------------|
 * | \ref Segments_findInAllSegments | All segments | Searches all segments in the container |
 * | \ref Segments_findInSegments_range | Segments [begin, end) | Searches segments in a specified range |
 * | \ref Segments_findInSegments_with_segment_indices | Segments in array | Searches only segments whose indices are in the
 * provided array |
 *
 * **When to use:**
 * - Use `findInAllSegments` when you need to search through all segments
 * - Use `findInSegments` with range when you want to search a contiguous range of segments
 * - Use `findInSegments` with array when you have a specific, non-contiguous set of segment indices
 *
 * \subsection SegmentFindConditionalFunctions Conditional Find Functions
 *
 * These functions add an additional segment-level condition, searching only in segments that satisfy both
 * the segment condition and contain elements matching the element condition:
 *
 * | Function | Segments Searched | Description |
 * |----------|------------------|-------------|
 * | \ref Segments_findInAllSegmentsIf | All segments matching condition | Searches all segments that satisfy the segment
 * condition |
 * | \ref Segments_findInSegmentsIf | Segments [begin, end) matching condition | Searches segments in range that
 * satisfy the segment condition |
 *
 * Note: The segment condition allows to skip entire segments based on segment-level properties, and so to improve
 * performance.
 *
 * \section SegmentFindParameters Common Parameters
 *
 * All find functions share these common parameters:
 *
 * - **segments**: The segments container to search in
 * - **condition**: Lambda that tests if an element matches the search criteria (see \ref SegmentFindConditionLambda)
 * - **keeper**: Lambda that processes the search results for each segment (see \ref SegmentFindResultKeeperLambda)
 * - **launchConfig**: Configuration for parallel execution (optional)
 *
 * Conditional variants additionally require:
 * - **segmentCondition**: Lambda that determines if a segment should be searched (see \ref SegmentFindSegmentConditionLambda)
 *
 * \section SegmentFindUsageGuidelines Usage Guidelines
 *
 * **Performance considerations:**
 * - Use segment-level conditions (`*If` variants) to avoid unnecessary work on segments you don't need to search.
 * - The array overload is useful when you have pre-computed which segments to search. It also skips segments that are not
 *   needed and thus can improve performance.
 *
 * \section SegmentFindRelatedPages Related Pages
 *
 * - \ref SegmentFindLambdas - Detailed lambda function signatures
 */

/**
 * \page SegmentFindLambdas Lambda Functions for Segment Find Operations
 *
 * \tableofcontents
 *
 * This page describes the lambda function signatures used in segment find operations.
 *
 * \section SegmentFindConditionLambda Condition Lambda
 *
 * The condition lambda determines whether an element within a segment satisfies the search condition.
 * It has the following signature:
 *
 * ```cpp
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> bool
 * {
 *    // Return true if element satisfies the condition
 *    return ...;
 * };
 * ```
 *
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * The lambda should return `true` if the element satisfies the search condition.
 *
 * \section SegmentFindResultKeeperLambda Result Keeper Lambda
 *
 * The result keeper lambda manages the results of the search operation.
 * It has the following signature:
 *
 * ```cpp
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, bool found )
 * {
 *    // Process the search result
 *    if( found ) {
 *       // localIdx points to the position where element was found
 *    }
 * };
 * ```
 *
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the index of the element within the segment (valid only if found is true).
 * - \e found is a boolean indicating whether the element was found.
 *
 * This lambda is called for each processed segment. If `found` is true, `localIdx` points at the position
 * in the segment where the element was found.
 *
 * \section SegmentFindSegmentConditionLambda Segment Condition Lambda
 *
 * The segment condition lambda determines which segments should be searched.
 * It has the following signature:
 *
 * ```cpp
 * auto segmentCondition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool
 * {
 *    // Return true if segment should be searched
 *    return ...;
 * };
 * ```
 *
 * - \e segmentIdx is the index of the segment.
 *
 * The lambda should return `true` if the segment should be searched.
 */

namespace TNL::Algorithms::Segments {

/**
 * \brief In each segment, find the first occurrence of an element fulfilling specified condition.
 * \anchor Segments_findInAllSegments
 *
 * \tparam Segments is the type of the segments.
 * \tparam Condition is the type of the lambda function expressing the condition.
 * \tparam ResultKeeper is the type of the lambda function that will manage the results of searching.
 *
 * \param segments is the segments to search in.
 * \param condition is the lambda function returning true for the found element. See \ref SegmentFindConditionLambda.
 * \param keeper is the lambda function managing the results of the searching. See \ref SegmentFindResultKeeperLambda.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_find.cpp
 *
 * \par Output
 * \include SegmentsExample_find.out
 *
 * Note: A function like `find` searching for a specific value does not make sense for segments due
 * to the necessity of accessing the data via a lambda function anyway.
 */
template< typename Segments, typename Condition, typename ResultKeeper >
static void
findInAllSegments( const Segments& segments,
                   Condition&& condition,
                   ResultKeeper&& keeper,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief In each segment of given range, find the first occurrence of an element fulfilling specified condition.
 * \anchor Segments_findInSegments_range
 *
 * \tparam Segments is the type of the segments.
 * \tparam IndexBegin is the type of the index defining the range of segments to search in.
 * \tparam IndexEnd is the type of the index defining the range of segments to search in.
 * \tparam Condition is the type of the lambda function expressing the condition.
 * \tparam ResultKeeper is the type of the lambda function that will manage the results of searching.
 *
 * \param segments is the segments to search in.
 * \param begin defines the range [begin,end) of segments to search in.
 * \param end defines the range [begin,end) of segments to search in.
 * \param condition is the lambda function returning true for the found element. See \ref SegmentFindConditionLambda.
 * \param keeper is the lambda function managing the results of the searching. See \ref SegmentFindResultKeeperLambda.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_find.cpp
 *
 * \par Output
 * \include SegmentsExample_find.out
 *
 * Note: A function like `find` searching for a specific value does not make sense for segments due
 * to the necessity of accessing the data via a lambda function anyway.
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename ResultKeeper,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
findInSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Condition&& condition,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief In each segment within given segment indexes, find the first occurrence of an element fulfilling specified condition.
 * \anchor Segments_findInSegments_with_segment_indices
 *
 * \tparam Segments is the type of the segments.
 * \tparam Array is the type of the array holding the segment indexes.
 * \tparam Condition is the type of the lambda function expressing the condition.
 * \tparam ResultKeeper is the type of the lambda function that will manage the results of searching.
 *
 * \param segments is the segments to search in.
 * \param segmentIndexes is the array holding the indexes of segments to search in.
 * \param condition is the lambda function returning true for the found element. See \ref SegmentFindConditionLambda.
 * \param keeper is the lambda function managing the results of the searching. See \ref SegmentFindResultKeeperLambda.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_find.cpp
 *
 * \par Output
 * \include SegmentsExample_find.out
 *
 * Note: A function like `find` searching for a specific value does not make sense for segments due
 * to the necessity of accessing the data via a lambda function anyway.
 */
template< typename Segments,
          typename Array,
          typename Condition,
          typename ResultKeeper,
          typename T = std::enable_if_t< IsArrayType< Array >::value > >
static void
findInSegments( const Segments& segments,
                const Array& segmentIndexes,
                Condition&& condition,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief In each segment fulfilling specified segment condition, find the first occurrence of an element fulfilling specified
 * element condition.
 * \anchor Segments_findInAllSegmentsIf
 *
 * \tparam Segments is the type of the segments.
 * \tparam SegmentCondition is the type lambda function masking the segments to search in.
 * \tparam Condition is the type of the lambda function expressing the condition.
 * \tparam ResultKeeper is the type of the lambda function that will manage the results of searching.
 *
 * \param segments is the segments to search in.
 * \param segmentCondition is the lambda function returning true for the segments to search in. See \ref
 * SegmentFindSegmentConditionLambda.
 * \param condition is the lambda function returning true for the found element. See \ref SegmentFindConditionLambda.
 * \param keeper is the lambda function managing the results of the searching. See \ref SegmentFindResultKeeperLambda.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_find.cpp
 *
 * \par Output
 * \include SegmentsExample_find.out
 *
 * Note: A function like `find` searching for a specific value does not make sense for segments due
 * to the necessity of accessing the data via a lambda function anyway.
 */
template< typename Segments, typename SegmentCondition, typename Condition, typename ResultKeeper >
static void
findInAllSegmentsIf( const Segments& segments,
                     SegmentCondition&& segmentCondition,
                     Condition&& condition,
                     ResultKeeper&& keeper,
                     LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief In each segment within given range and fulfilling specified segment condition, find the first occurrence of an element
 * fulfilling specified element condition.
 * \anchor Segments_findInSegmentsIf
 *
 * \tparam Segments is the type of the segments.
 * \tparam IndexBegin is the type of the index defining the range of segment indexes to search in.
 * \tparam IndexEnd is the type of the index defining the range of segments indexes to search in.
 * \tparam SegmentCondition is the type lambda function masking the segments to search in.
 * \tparam Condition is the type of the lambda function expressing the condition.
 * \tparam ResultKeeper is the type of the lambda function that will manage the results of searching.
 *
 * \param segments is the segments to search in.
 * \param begin defines the range [begin,end) of segments to search in.
 * \param end defines the range [begin,end) of segments to search in.
 * \param segmentCondition is the lambda function returning true for the segments to search in. See \ref
 * SegmentFindSegmentConditionLambda.
 * \param condition is the lambda function returning true for the found element. See \ref SegmentFindConditionLambda.
 * \param keeper is the lambda function managing the results of the searching. See \ref SegmentFindResultKeeperLambda.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_find.cpp
 *
 * \par Output
 * \include SegmentsExample_find.out
 *
 * Note: A function like `find` searching for a specific value does not make sense for segments due
 * to the necessity of accessing the data via a lambda function anyway.
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename SegmentCondition,
          typename Condition,
          typename ResultKeeper,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
findInSegmentsIf( const Segments& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  SegmentCondition&& segmentCondition,
                  Condition&& condition,
                  ResultKeeper&& keeper,
                  LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  //namespace TNL::Algorithms::Segments

#include "find.hpp"
