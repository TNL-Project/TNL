// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "LaunchConfiguration.h"
#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::Segments {

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                const Value& identity,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegments( const Segments& segments,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   const Value& identity,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegments( const Segments& segments,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceSegmentsWithArg( const Segments& segments,
                       IndexBegin begin,
                       IndexEnd end,
                       Fetch&& fetch,
                       Reduction&& reduction,
                       ResultKeeper&& keeper,
                       const Value& identity,
                       LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegmentsWithArgument( const Segments& segments,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultKeeper&& keeper,
                               const Value& identity,
                               LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegmentsWithArgument( const Segments& segments,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultKeeper&& keeper,
                               LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Algorithms::Segments

#include "reduce.hpp"
