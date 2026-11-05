// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/TypeTraits.h>
#include "detail/MultiReducingOperations.h"
#include "multiReduce.h"

namespace TNL::Algorithms::Segments {

template< int Size, typename Segments, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
static void
multiReduceAllSegments(
   const Segments& segments,
   Fetch&& fetch,
   Reduction&& reduction,
   ResultStorer&& storer,
   const Value& identity,
   LaunchConfiguration launchConfig )
{
   static_assert( Size > 0, "The number of reductions must be positive." );
   static_assert(
      callableArgumentCount< std::decay_t< Fetch > >() == 4 || callableArgumentCount< std::decay_t< Fetch > >() == 2,
      "The fetch lambda of the multireduction must take either (segmentIdx, localIdx, globalIdx, j) "
      "or (globalIdx, j)." );
   detail::MultiReducingOperations< typename Segments::ConstViewType >::template multiReduceAllSegments< Size >(
      segments.getConstView(),
      std::forward< Fetch >( fetch ),
      std::forward< Reduction >( reduction ),
      std::forward< ResultStorer >( storer ),
      identity,
      launchConfig );
}

template< int Size, typename Segments, typename Fetch, typename Reduction, typename ResultStorer >
static void
multiReduceAllSegments(
   const Segments& segments,
   Fetch&& fetch,
   Reduction&& reduction,
   ResultStorer&& storer,
   LaunchConfiguration launchConfig )
{
   using ValueType = detail::MultiReductionFetchValue< typename Segments::IndexType, std::decay_t< Fetch > >;
   multiReduceAllSegments< Size >(
      segments,
      std::forward< Fetch >( fetch ),
      std::forward< Reduction >( reduction ),
      std::forward< ResultStorer >( storer ),
      std::decay_t< Reduction >::template getIdentity< ValueType >(),
      launchConfig );
}

template< int TileSize, typename Segments, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
static void
multiReduceAllSegmentsInTiles(
   const Segments& segments,
   int reductions,
   Fetch&& fetch,
   Reduction&& reduction,
   ResultStorer&& storer,
   const Value& identity,
   LaunchConfiguration launchConfig )
{
   static_assert( TileSize > 0, "The tile size must be positive." );
   static_assert(
      callableArgumentCount< std::decay_t< Fetch > >() == 4 || callableArgumentCount< std::decay_t< Fetch > >() == 2,
      "The fetch lambda of the multireduction must take either (segmentIdx, localIdx, globalIdx, j) "
      "or (globalIdx, j)." );
   if( reductions <= 0 )
      throw std::invalid_argument( "multiReduceAllSegmentsInTiles: the number of reductions must be positive" );
   detail::MultiReducingOperations< typename Segments::ConstViewType >::template multiReduceAllSegmentsInTiles< TileSize >(
      segments.getConstView(),
      reductions,
      std::forward< Fetch >( fetch ),
      std::forward< Reduction >( reduction ),
      std::forward< ResultStorer >( storer ),
      identity,
      launchConfig );
}

template<
   int Lanes,
   int ReductionsPerLane,
   typename Segments,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
static void
multiReduceAllSegmentsLanesOverReductions(
   const Segments& segments,
   int reductions,
   Fetch&& fetch,
   Reduction&& reduction,
   ResultStorer&& storer,
   const Value& identity,
   typename Segments::IndexType maxSegmentSize )
{
   static_assert(
      Lanes > 0 && ( Lanes & ( Lanes - 1 ) ) == 0 && Lanes <= Backend::getMinWarpSize(),
      "The number of lanes must be a power of two up to the warp size." );
   static_assert( ReductionsPerLane > 0, "The number of reductions per lane must be positive." );
   static_assert(
      callableArgumentCount< std::decay_t< Fetch > >() == 4 || callableArgumentCount< std::decay_t< Fetch > >() == 2,
      "The fetch lambda of the multireduction must take either (segmentIdx, localIdx, globalIdx, j) "
      "or (globalIdx, j)." );
   if( reductions <= 0 )
      throw std::invalid_argument( "multiReduceAllSegmentsLanesOverReductions: the number of reductions must be positive" );
   detail::MultiReducingOperations< typename Segments::ConstViewType >::
      template multiReduceAllSegmentsLanesOverReductions< Lanes, ReductionsPerLane >(
         segments.getConstView(),
         reductions,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< ResultStorer >( storer ),
         identity,
         maxSegmentSize );
}

}  // namespace TNL::Algorithms::Segments
