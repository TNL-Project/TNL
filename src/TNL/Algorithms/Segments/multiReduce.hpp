// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <TNL/TypeTraits.h>
#include "multiReduce.h"
#include "reduce.h"

namespace TNL::Algorithms::Segments {

namespace detail {

/**
 * \brief Partial results of `Size` reductions processed together by \ref Segments_multiReduceAllSegments.
 *
 * Unlike \ref TNL::Containers::StaticArray, it is trivially copyable for trivially copyable
 * `Value`, so the reduction kernels can exchange it by warp shuffles.
 */
template< int Size, typename Value >
struct MultiReductionValue
{
   Value values[ Size ];

   __cuda_callable__
   Value&
   operator[]( int j )
   {
      return values[ j ];
   }

   __cuda_callable__
   const Value&
   operator[]( int j ) const
   {
      return values[ j ];
   }
};

// Fetches the values of all reductions for the full form of the fetch lambda.
template< int Size, typename Index, typename Fetch, typename Value >
struct MultiReductionFetch
{
   mutable Fetch fetch;

   __cuda_callable__
   MultiReductionValue< Size, Value >
   operator()( Index segmentIdx, Index localIdx, Index globalIdx ) const
   {
      MultiReductionValue< Size, Value > result;
      for( int j = 0; j < Size; j++ )
         result[ j ] = fetch( segmentIdx, localIdx, globalIdx, j );
      return result;
   }
};

// Fetches the values of all reductions for the brief form of the fetch lambda.
template< int Size, typename Index, typename Fetch, typename Value >
struct MultiReductionBriefFetch
{
   mutable Fetch fetch;

   __cuda_callable__
   MultiReductionValue< Size, Value >
   operator()( Index globalIdx ) const
   {
      MultiReductionValue< Size, Value > result;
      for( int j = 0; j < Size; j++ )
         result[ j ] = fetch( globalIdx, j );
      return result;
   }
};

// Applies the reduction to each of the partial results.
template< int Size, typename Reduction, typename Value >
struct MultiReductionOperation
{
   mutable Reduction reduction;

   __cuda_callable__
   MultiReductionValue< Size, Value >
   operator()( const MultiReductionValue< Size, Value >& a, const MultiReductionValue< Size, Value >& b ) const
   {
      MultiReductionValue< Size, Value > result;
      for( int j = 0; j < Size; j++ )
         result[ j ] = reduction( a[ j ], b[ j ] );
      return result;
   }
};

// Passes the results of all reductions to the storer.
template< int Size, typename Index, typename ResultStorer, typename Value >
struct MultiReductionStorer
{
   mutable ResultStorer storer;

   __cuda_callable__
   void
   operator()( Index segmentIdx, const MultiReductionValue< Size, Value >& value ) const
   {
      for( int j = 0; j < Size; j++ )
         storer( segmentIdx, j, value[ j ] );
   }
};

// Return type of the fetch lambda of the multireduction.
template< typename Index, typename Fetch >
using MultiReductionFetchValue = std::decay_t< typename std::conditional_t<
   callableArgumentCount< Fetch >() == 4,
   std::invoke_result< Fetch, Index, Index, Index, int >,
   std::invoke_result< Fetch, Index, int > >::type >;

}  // namespace detail

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
   using IndexType = typename Segments::IndexType;
   using FetchType = std::decay_t< Fetch >;
   using ValueType = detail::MultiReductionFetchValue< IndexType, FetchType >;
   using MultiValue = detail::MultiReductionValue< Size, ValueType >;
   static_assert(
      callableArgumentCount< FetchType >() == 4 || callableArgumentCount< FetchType >() == 2,
      "The fetch lambda of the multireduction must take either (segmentIdx, localIdx, globalIdx, j) "
      "or (globalIdx, j)." );

   MultiValue multiIdentity;
   for( int j = 0; j < Size; j++ )
      multiIdentity[ j ] = identity;
   detail::MultiReductionOperation< Size, std::decay_t< Reduction >, ValueType > multiReduction{ reduction };
   detail::MultiReductionStorer< Size, IndexType, std::decay_t< ResultStorer >, ValueType > multiStorer{ storer };

   if constexpr( callableArgumentCount< FetchType >() == 4 ) {
      detail::MultiReductionFetch< Size, IndexType, FetchType, ValueType > multiFetch{ fetch };
      reduceAllSegments( segments, multiFetch, multiReduction, multiStorer, multiIdentity, launchConfig );
   }
   else {
      detail::MultiReductionBriefFetch< Size, IndexType, FetchType, ValueType > multiFetch{ fetch };
      reduceAllSegments( segments, multiFetch, multiReduction, multiStorer, multiIdentity, launchConfig );
   }
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

}  // namespace TNL::Algorithms::Segments
