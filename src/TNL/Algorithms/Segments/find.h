// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"
#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief In each segment, find the first occurrence of an element fulfilling specified condition.
 *
 * \tparam Segments is the type of the segments.
 * \tparam Condition is the type of the lambda function expressing the condition.
 * \tparam ResultKeeper is the type of the lambda function that will manage the results of searching.
 *
 * \param segments is the segments to search in.
 * \param condition is the lambda function returning true for the found element.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) ->
 * bool`.
 * \param keeper is the lambda function managing the results of the searching.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, bool found )`.
 *    Here, `segmentIdx` is the index of the segment, `localIdx` is the index of the element within the segment,
 *    and `found` is a boolean indicating whether the element was found. This lambda function is called for
 *    each segment. If `found` is true, `localIdx` points at the position in the segment where the element was found.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_find.cpp
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
 * \param condition is the lambda function returning true for the found element.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) ->
 * bool`.
 * \param keeper is the lambda function managing the results of the searching.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, bool found )`.
 *    Here, `segmentIdx` is the index of the segment, `localIdx` is the index of the element within the segment,
 *    and `found` is a boolean indicating whether the element was found. This lambda function is called for
 *    each segment within the range [begin, end). If `found` is true, `localIdx` points at the position
 *    in the segment where the element was found.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_find.cpp
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
 *
 * \tparam Segments is the type of the segments.
 * \tparam Array is the type of the array holding the segment indexes.
 * \tparam Condition is the type of the lambda function expressing the condition.
 * \tparam ResultKeeper is the type of the lambda function that will manage the results of searching.
 *
 * \param segments is the segments to search in.
 * \param segmentIndexes is the array holding the indexes of segments to search in.
 * \param condition is the lambda function returning true for the found element.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) ->
 * bool`.
 * \param keeper is the lambda function managing the results of the searching.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, bool found )`.
 *    Here, `segmentIdx` is the index of the segment, `localIdx` is the index of the element within the segment,
 *    and `found` is a boolean indicating whether the element was found. This lambda function is called for
 *    each segment index of given segment indexes. If `found` is true, `localIdx`
 *    points at the position in the segment where the element was found.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_find.cpp
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
 *
 * \tparam Segments is the type of the segments.
 * \tparam SegmentCondition is the type lambda function masking the segments to search in.
 * \tparam Condition is the type of the lambda function expressing the condition.
 * \tparam ResultKeeper is the type of the lambda function that will manage the results of searching.
 *
 * \param segments is the segments to search in.
 * \param segmentCondition is the lambda function returning true for the segments to search in.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx ) -> bool`.
 * \param condition is the lambda function returning true for the found element.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) ->
 * bool`.
 * \param keeper is the lambda function managing the results of the searching.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, bool found )`.
 *    Here, `segmentIdx` is the index of the segment, `localIdx` is the index of the element within the segment,
 *    and `found` is a boolean indicating whether the element was found. This lambda function is called for
 *    each segment index of given segment indexes. If `found` is true, `localIdx`
 *    points at the position in the segment where the element was found.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_find.cpp
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
 * \param segmentCondition is the lambda function returning true for the segments to search in.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx ) -> bool`.
 * \param condition is the lambda function returning true for the found element.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) ->
 * bool`.
 * \param keeper is the lambda function managing the results of the searching.
 *    It should have signature `[=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, bool found )`.
 *    Here, `segmentIdx` is the index of the segment, `localIdx` is the index of the element within the segment,
 *    and `found` is a boolean indicating whether the element was found. This lambda function is called for
 *    each segment index of given segment indexes. If `found` is true, `localIdx`
 *    points at the position in the segment where the element was found.
 * \param launchConfig is the configuration for launching the kernel.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_find.cpp
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
