// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Math.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Segments::detail {

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

// The full form of the fetch lambda for the reductions first + j.
template< typename Index, typename Fetch >
struct MultiReductionShiftedFetch
{
   mutable Fetch fetch;
   int first;

   __cuda_callable__
   auto
   operator()( Index segmentIdx, Index localIdx, Index globalIdx, int j ) const
   {
      return fetch( segmentIdx, localIdx, globalIdx, first + j );
   }
};

// The brief form of the fetch lambda for the reductions first + j.
template< typename Index, typename Fetch >
struct MultiReductionShiftedBriefFetch
{
   mutable Fetch fetch;
   int first;

   __cuda_callable__
   auto
   operator()( Index globalIdx, int j ) const
   {
      return fetch( globalIdx, first + j );
   }
};

// The storer of the reductions first + j.
template< typename Index, typename ResultStorer, typename Value >
struct MultiReductionShiftedStorer
{
   mutable ResultStorer storer;
   int first;

   __cuda_callable__
   void
   operator()( Index segmentIdx, int j, const Value& value ) const
   {
      storer( segmentIdx, first + j, value );
   }
};

// Return type of the fetch lambda of the multireduction.
template< typename Index, typename Fetch >
using MultiReductionFetchValue = std::decay_t< typename std::conditional_t<
   callableArgumentCount< Fetch >() == 4,
   std::invoke_result< Fetch, Index, Index, Index, int >,
   std::invoke_result< Fetch, Index, int > >::type >;

// Calls f( std::integral_constant< int, size >{} ) for size in [1, MaxSize].
template< int MaxSize, typename Function >
void
dispatchMultiReductionSize( int size, Function&& f )
{
   if constexpr( MaxSize > 0 ) {
      if( size == MaxSize )
         f( std::integral_constant< int, MaxSize >{} );
      else
         dispatchMultiReductionSize< MaxSize - 1 >( size, std::forward< Function >( f ) );
   }
}

/**
 * \brief Implementation of the multireduction in segments.
 *
 * The specializations of \ref MultiReducingOperations for the segment types inherit the general
 * implementations from this class and override those for which they have a dedicated one.
 */
template< typename Segments >
struct MultiReducingOperations;

template< typename Segments >
struct MultiReducingOperationsBase
{
   using IndexType = typename Segments::IndexType;

   /**
    * \brief General implementation of \ref Segments_multiReduceAllSegments_identity.
    *
    * It calls \ref Segments_reduceAllSegments with a trivially copyable aggregate of the partial
    * results of all reductions, so it works with all segment types and launch configurations.
    */
   template< int Size, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   multiReduceAllSegments(
      const Segments& segments,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using FetchType = std::decay_t< Fetch >;
      using ValueType = MultiReductionFetchValue< IndexType, FetchType >;
      using MultiValue = MultiReductionValue< Size, ValueType >;

      MultiValue multiIdentity;
      for( int j = 0; j < Size; j++ )
         multiIdentity[ j ] = identity;
      MultiReductionOperation< Size, std::decay_t< Reduction >, ValueType > multiReduction{ reduction };
      MultiReductionStorer< Size, IndexType, std::decay_t< ResultStorer >, ValueType > multiStorer{ storer };

      if constexpr( callableArgumentCount< FetchType >() == 4 ) {
         MultiReductionFetch< Size, IndexType, FetchType, ValueType > multiFetch{ fetch };
         reduceAllSegments( segments, multiFetch, multiReduction, multiStorer, multiIdentity, launchConfig );
      }
      else {
         MultiReductionBriefFetch< Size, IndexType, FetchType, ValueType > multiFetch{ fetch };
         reduceAllSegments( segments, multiFetch, multiReduction, multiStorer, multiIdentity, launchConfig );
      }
   }

   /**
    * \brief General implementation of \ref TNL::Algorithms::Segments::multiReduceAllSegmentsInTiles.
    *
    * The tiles of `TileSize` reductions are processed one after another by \ref multiReduceAllSegments.
    */
   template< int TileSize, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
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
      using FetchType = std::decay_t< Fetch >;
      using ValueType = MultiReductionFetchValue< IndexType, FetchType >;
      for( int first = 0; first < reductions; first += TileSize ) {
         const MultiReductionShiftedStorer< IndexType, std::decay_t< ResultStorer >, ValueType > shiftedStorer{ storer, first };
         auto run = [ & ]( auto size )
         {
            constexpr int Size = decltype( size )::value;
            if constexpr( callableArgumentCount< FetchType >() == 4 )
               multiReduceAllSegments< Size >(
                  segments,
                  MultiReductionShiftedFetch< IndexType, FetchType >{ fetch, first },
                  reduction,
                  shiftedStorer,
                  identity,
                  launchConfig );
            else
               multiReduceAllSegments< Size >(
                  segments,
                  MultiReductionShiftedBriefFetch< IndexType, FetchType >{ fetch, first },
                  reduction,
                  shiftedStorer,
                  identity,
                  launchConfig );
         };
         dispatchMultiReductionSize< TileSize >( TNL::min( reductions - first, TileSize ), run );
      }
   }

   /**
    * \brief General implementation of \ref TNL::Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions.
    *
    * The reductions are computed by the general multireduction, `ReductionsPerLane` reductions at
    * once. The threads are not distributed among the reductions and `maxSegmentSize` is not used.
    */
   template< int Lanes, int ReductionsPerLane, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   multiReduceAllSegmentsLanesOverReductions(
      const Segments& segments,
      int reductions,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      IndexType maxSegmentSize )
   {
      multiReduceAllSegmentsInTiles< ReductionsPerLane >(
         segments,
         reductions,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< ResultStorer >( storer ),
         identity,
         LaunchConfiguration() );
   }
};

}  // namespace TNL::Algorithms::Segments::detail
