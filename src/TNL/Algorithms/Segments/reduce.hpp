// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "reduce.h"
#include "detail/ReducingOperations.h"

namespace TNL::Algorithms::Segments {

template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
static void
reduceSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                const Value& identity,
                LaunchConfiguration launchConfig )
{
   detail::ReducingOperations< typename Segments::ConstViewType >::reduceSegments( segments.getConstView(),
                                                                                   begin,
                                                                                   end,
                                                                                   std::forward< Fetch >( fetch ),
                                                                                   std::forward< Reduction >( reduction ),
                                                                                   std::forward< ResultKeeper >( keeper ),
                                                                                   identity,
                                                                                   launchConfig );
}

template< typename Segments, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
static void
reduceAllSegments( const Segments& segments,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   const Value& identity,
                   LaunchConfiguration launchConfig )
{
   using IndexType = typename Segments::IndexType;
   reduceSegments( segments,
                   (IndexType) 0,
                   segments.getSegmentsCount(),
                   std::forward< Fetch >( fetch ),
                   std::forward< Reduction >( reduction ),
                   std::forward< ResultKeeper >( keeper ),
                   identity,
                   launchConfig );
}

}  // namespace TNL::Algorithms::Segments
