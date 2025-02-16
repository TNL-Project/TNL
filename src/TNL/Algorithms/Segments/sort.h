// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"

namespace TNL::Algorithms::Segments {

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

template< typename Segments, typename Fetch, typename Compare, typename Swap >
static void
sortAllSegments( const Segments& segments,
                 Fetch&& fetch,
                 Compare&& compare,
                 Swap&& swap,
                 LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

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

template< typename Segments, typename Condition, typename Fetch, typename Compare, typename Swap >
static void
sortAllSegmentsIf( const Segments& segments,
                   Condition&& condition,
                   Fetch&& fetch,
                   Compare&& compare,
                   Swap&& swap,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Algorithms::Segments

#include "sort.hpp"