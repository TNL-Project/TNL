// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"
#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Find the first occurrence of a value in segments.
 *
 * \tparam Segments is the type of the segments.
 * \tparam IndexBegin is the type of the index of the first segment.
 * \tparam IndexEnd is the type of the index of the last segment.
 *
 * Note, that function like `find` searching for a specific value does not make sense for segments due
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

template< typename Segments, typename Condition, typename ResultKeeper >
static void
findInAllSegments( const Segments& segments,
                   Condition&& condition,
                   ResultKeeper&& keeper,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename ResultKeeper,
          typename T = std::enable_if_t< IsArrayType< Array >::value
                                         && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
findInSegments( const Segments& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
                Condition&& condition,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

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

template< typename Segments, typename SegmentCondition, typename Condition, typename ResultKeeper >
static void
findInSegmentsIf( const Segments& segments,
                  SegmentCondition&& segmentCondition,
                  Condition&& condition,
                  ResultKeeper&& keeper,
                  LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  //namespace TNL::Algorithms::Segments

#include "find.hpp"