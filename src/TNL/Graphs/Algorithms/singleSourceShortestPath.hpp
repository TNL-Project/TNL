// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <queue>
#include <stdexcept>
#include <type_traits>

#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>
#include <TNL/Atomic.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

#include "details/activeVertices.hpp"
#include "details/parallelTraversal.hpp"
#include "singleSourceShortestPath.h"

namespace TNL::Graphs::Algorithms {

namespace detail {

template< typename EdgeWeightCallable, typename Graph >
struct IsSsspEdgeWeightCallable : std::bool_constant< std::is_invocable_r_v<
                                     typename Graph::ValueType,
                                     EdgeWeightCallable,
                                     typename Graph::IndexType,
                                     typename Graph::IndexType,
                                     typename Graph::ValueType > >
{};

template< typename EdgeWeightCallable, typename Graph >
constexpr bool isSsspEdgeWeightCallable_v = IsSsspEdgeWeightCallable< EdgeWeightCallable, Graph >::value;

template< typename Real >
__cuda_callable__
bool
isBlockedSsspEdgeWeight( const Real& weight )
{
   // Returning +/- infinity from the edge-weight callable signals that the
   // edge is non-traversable (treated as if it does not exist).
   return weight == std::numeric_limits< Real >::infinity() || weight == -std::numeric_limits< Real >::infinity();
}

}  // namespace detail

template<
   typename Graph,
   typename Vector,
   typename ActivePredicate,
   typename EdgeWeightCallable,
   typename IndexType = typename Graph::IndexType >
void
parallelSingleSourceShortestPath(
   const Graph& graph,
   IndexType start,
   ActivePredicate&& isActive,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   const IndexType n = graph.getVertexCount();
   distances.setSize( n );

   // Bellman-Ford-style parallel relaxation: each iteration processes the
   // current frontier and relaxes all outgoing edges.  A vertex enters the
   // next frontier when its distance was improved in this round.
   //
   // y            – working copy of distances (updated concurrently)
   // predecessors – parent vertex for each visited node
   // marks        – 1 if the vertex was improved in this iteration, 0 otherwise
   // marksScan    – inclusive prefix sum of marks (compacts the next frontier)
   // frontier     – dense array of vertex indices forming the current frontier
   Vector y( distances.getSize() );
   Containers::Vector< IndexType, DeviceType, IndexType > predecessors( n, -1 );
   Containers::Vector< IndexType, DeviceType, IndexType > marks( n );
   Containers::Vector< IndexType, DeviceType, IndexType > marksScan( n, 0 );
   Containers::Vector< IndexType, DeviceType, IndexType > frontier( n, 0 );
   distances = std::numeric_limits< ValueType >::max();
   distances.setElement( start, 0 );
   frontier.setElement( 0, start );
   IndexType frontierSize( 1 );
   y = distances;
   auto yView = y.getView();
   auto predecessorsView = predecessors.getView();
   auto marksView = marks.getView();

   // On Host we need an atomic copy of y to avoid the check-then-write race
   // when multiple OpenMP threads relax the same target vertex concurrently.
   using HostAtomicRealVec = Containers::Vector< Atomic< ValueType, Devices::Host >, Devices::Host, IndexType >;
   HostAtomicRealVec hostAtomicY;
   if constexpr( std::is_same_v< DeviceType, Devices::Host > )
      hostAtomicY.setSize( n );

   for( IndexType i = 0; i < n; i++ ) {
      marks = 0;
      if constexpr( std::is_same_v< DeviceType, Devices::Host > ) {
         // Copy current distances into the atomic buffer
         auto hostAtomicYView = hostAtomicY.getView();
         TNL::Algorithms::parallelFor< DeviceType >(
            0,
            n,
            [ = ] __cuda_callable__( IndexType idx ) mutable
            {
               hostAtomicYView[ idx ] = yView[ idx ];
            } );

         forEdges(
            graph,
            frontier,
            0,
            frontierSize,
            [ = ] __cuda_callable__(
               IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
            {
               if( targetIdx != Matrices::paddingIndex< IndexType > && isActive( targetIdx ) ) {
                  const ValueType transformedWeight = edgeWeightCallable( sourceIdx, targetIdx, weight );
                  if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
                     return;

                  ValueType newDistance = yView[ sourceIdx ] + transformedWeight;
                  // Atomically reduce: only update if newDistance is smaller.
                  // fetch_min returns the old value; if it was larger, our
                  // update took effect and we record the predecessor.
                  const ValueType oldDistance = hostAtomicYView[ targetIdx ].fetch_min( newDistance );
                  if( newDistance < oldDistance ) {
                  // The predecessor may be overwritten by a concurrent
                  // thread that achieves an even shorter distance — this
                  // is benign: the distance is always correct, and the
                  // predecessor will be fixed in a subsequent iteration.
#if defined( HAVE_OPENMP )
   #pragma omp atomic write
#endif
                     predecessorsView[ targetIdx ] = sourceIdx;
#if defined( HAVE_OPENMP )
   #pragma omp atomic write
#endif
                     marksView[ targetIdx ] = 1;
                  }
               }
            },
            launchConfig );

         // Copy atomic results back to y
         TNL::Algorithms::parallelFor< DeviceType >(
            0,
            n,
            [ = ] __cuda_callable__( IndexType idx ) mutable
            {
               yView[ idx ] = hostAtomicYView[ idx ].load();
            } );
      }
      else  // if constexpr( std::is_same_v< DeviceType, Devices::Host > )
         forEdges(
            graph,
            frontier,
            0,
            frontierSize,
            [ = ] __cuda_callable__(
               IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
            {
               TNL_ASSERT_GE( sourceIdx, 0, "" );
               TNL_ASSERT_LT( sourceIdx, yView.getSize(), "" );
               TNL_ASSERT_GE( targetIdx, 0, "" );
               TNL_ASSERT_LT( targetIdx, yView.getSize(), "" );
               if( targetIdx != Matrices::paddingIndex< IndexType > && isActive( targetIdx ) ) {
                  const ValueType transformedWeight = edgeWeightCallable( sourceIdx, targetIdx, weight );
                  if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
                     return;

                  ValueType newDistance = yView[ sourceIdx ] + transformedWeight;
                  if( newDistance < yView[ targetIdx ] ) {
                     atomicMin( &yView[ targetIdx ], newDistance );
                     // The predecessor is set to the *smallest* source index
                     // among concurrent relaxers, not necessarily the one that
                     // provided the shortest path.  The distance itself is
                     // always correct (atomicMin guarantees the minimum).
                     atomicMin( &predecessorsView[ targetIdx ], sourceIdx );
                     atomicMax( &marksView[ targetIdx ], 1 );
                  }
               }
            },
            launchConfig );
      // Compact improved vertices into the next frontier
      frontierSize = detail::compactFrontier< DeviceType, IndexType >( marks, marksScan, frontier );
      if( frontierSize == 0 )
         break;
      distances = y;
   }
}

template< typename Graph, typename Vector, typename ActivePredicate, typename EdgeWeightCallable, typename Index >
void
singleSourceShortestPath_impl(
   const Graph& graph,
   Index start,
   ActivePredicate&& isActive,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert(
      ! Graph::AdjacencyMatrixType::MatrixType::isSymmetric(), "SSSP requires general adjacency matrix, not symmetric." );

   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;

   distances.setSize( graph.getVertexCount() );
   if( graph.getVertexCount() == 0 )
      return;
   TNL_ASSERT_GE( start, static_cast< Index >( 0 ), "Start vertex index must be non-negative." );
   TNL_ASSERT_LT( start, graph.getVertexCount(), "Start vertex index must be less than the number of vertices." );

   if( ! isActive( start ) )
      throw std::invalid_argument( "Start vertex must belong to the induced active subgraph." );

   distances = std::numeric_limits< ValueType >::max();
   distances.setElement( start, 0.0 );

   // Sequential backend: Dijkstra with a priority queue.
   if constexpr( std::is_same_v< DeviceType, TNL::Devices::Sequential > ) {
      // The priority queue stores pairs of (distance, vertex)
      std::priority_queue< std::pair< ValueType, Index >, std::vector< std::pair< ValueType, Index > >, std::greater<> > pq;
      pq.emplace( 0, start );

      while( ! pq.empty() ) {
         ValueType currentDistance;
         Index current;
         std::tie( currentDistance, current ) = pq.top();
         pq.pop();

         if( currentDistance > distances[ current ] )
            continue;

         const auto row = graph.getAdjacencyMatrix().getRow( current );
         for( Index i = 0; i < row.getSize(); i++ ) {
            const auto& edgeWeight = row.getValue( i );
            const auto& neighbor = row.getColumnIndex( i );
            if( neighbor == Matrices::paddingIndex< Index > )
               continue;
            if( ! isActive( neighbor ) )
               continue;

            const ValueType transformedWeight = edgeWeightCallable( current, neighbor, edgeWeight );
            if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
               continue;

            const ValueType distance = currentDistance + transformedWeight;

            if( distance < distances[ neighbor ] ) {
               distances[ neighbor ] = distance;
               // Lazy deletion: re-inserting a vertex already in the queue is
               // safe.  std::greater<> pops the smallest entry first, so the
               // guard above skips the stale larger copy (no decrease-key in
               // std::priority_queue).  O(E) pushes, standard for Dijkstra.
               pq.emplace( distance, neighbor );
            }
         }
      }
   }
   else {
      parallelSingleSourceShortestPath( graph, start, isActive, edgeWeightCallable, distances, launchConfig );
   }
   // Replace infinity sentinel with -1 for unreachable vertices
   distances.forAllElements(
      [] __cuda_callable__( Index i, ValueType & x )
      {
         x = ( x == std::numeric_limits< ValueType >::max() ) ? -1.0 : x;
      } );
}

template< typename Graph, typename Vector, typename Index >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   singleSourceShortestPath_impl(
      graph,
      start,
      [] __cuda_callable__( Index )
      {
         return true;
      },
      [] __cuda_callable__( Index, Index, typename Graph::ValueType weight )
      {
         return weight;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgeWeightCallable, typename Index, typename Enable >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert(
      detail::isSsspEdgeWeightCallable_v< EdgeWeightCallable, Graph >,
      "SSSP edge callable must return Graph::ValueType and accept (source, target) or (source, target, weight)." );

   singleSourceShortestPath_impl(
      graph,
      start,
      [] __cuda_callable__( Index )
      {
         return true;
      },
      std::forward< EdgeWeightCallable >( edgeWeightCallable ),
      distances,
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename Index, typename Enable >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   const VertexIndexes& vertexIndexes,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexVector = Containers::Vector< Index, DeviceType, Index >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   const auto isActive = [ = ] __cuda_callable__( Index vertex )
   {
      return static_cast< bool >( activeVerticesView[ vertex ] );
   };
   singleSourceShortestPath_impl(
      graph,
      start,
      isActive,
      [] __cuda_callable__( Index, Index, typename Graph::ValueType weight )
      {
         return weight;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename EdgeWeightCallable, typename Index, typename Enable >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   const VertexIndexes& vertexIndexes,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexVector = Containers::Vector< Index, DeviceType, Index >;

   static_assert(
      detail::isSsspEdgeWeightCallable_v< EdgeWeightCallable, Graph >,
      "SSSP edge callable must return Graph::ValueType and accept (source, target) or (source, target, weight)." );

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   const auto isActive = [ = ] __cuda_callable__( Index vertex )
   {
      return static_cast< bool >( activeVerticesView[ vertex ] );
   };
   singleSourceShortestPath_impl(
      graph, start, isActive, std::forward< EdgeWeightCallable >( edgeWeightCallable ), distances, launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename Index >
void
singleSourceShortestPathIf(
   const Graph& graph,
   Index start,
   VertexPredicate&& vertexPredicate,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   singleSourceShortestPath_impl(
      graph,
      start,
      predicate,
      [] __cuda_callable__( Index, Index, typename Graph::ValueType weight )
      {
         return weight;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgeWeightCallable, typename Index >
void
singleSourceShortestPathIf(
   const Graph& graph,
   Index start,
   VertexPredicate&& vertexPredicate,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert(
      detail::isSsspEdgeWeightCallable_v< EdgeWeightCallable, Graph >,
      "SSSP edge callable must return Graph::ValueType and accept (source, target) or (source, target, weight)." );

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   singleSourceShortestPath_impl(
      graph, start, predicate, std::forward< EdgeWeightCallable >( edgeWeightCallable ), distances, launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
