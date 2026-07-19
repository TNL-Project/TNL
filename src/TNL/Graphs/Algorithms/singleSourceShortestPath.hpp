// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <queue>
#include <stdexcept>
#include <type_traits>

#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>
#include <TNL/Atomic.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

#include "details/activeVertices.hpp"
#include "details/lambdaTraits.hpp"
#include "details/parallelTraversal.hpp"
#include "singleSourceShortestPath.h"

namespace TNL::Graphs::Algorithms {

namespace detail {

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

template< typename Graph, typename Vector, typename EdgeWeightCallable, typename IndexType = typename Graph::IndexType >
void
parallelSingleSourceShortestPath(
   const Graph& graph,
   IndexType start,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   double bitmapThreshold,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   const IndexType n = graph.getVertexCount();
   distances.setSize( n );

   Vector y( distances.getSize() );
   Containers::Vector< IndexType, DeviceType, IndexType > predecessors( n, -1 );
   Containers::Vector< IndexType, DeviceType, IndexType > marks( n );
   Containers::Vector< IndexType, DeviceType, IndexType > marksScan( n, 0 );
   Containers::Vector< IndexType, DeviceType, IndexType > frontier( n, 0 );
   Containers::Vector< IndexType, DeviceType, IndexType > nextMarks( n, 0 );
   distances = std::numeric_limits< ValueType >::max();
   distances.setElement( start, 0 );
   frontier.setElement( 0, start );
   IndexType frontierSize( 1 );
   y = distances;
   auto yView = y.getView();
   auto predecessorsView = predecessors.getView();
   auto marksView = marks.getView();

   marks = 0;
   marks.setElement( start, 1 );

   // On Host we need an atomic copy of y to avoid the check-then-write race
   // when multiple OpenMP threads relax the same target vertex concurrently.
   using HostAtomicRealVec = Containers::Vector< Atomic< ValueType, Devices::Host >, Devices::Host, IndexType >;
   HostAtomicRealVec hostAtomicY;
   if constexpr( std::is_same_v< DeviceType, Devices::Host > )
      hostAtomicY.setSize( n );

   for( IndexType i = 0; i < n; i++ ) {
      const double frontierFraction = static_cast< double >( frontierSize ) / static_cast< double >( n );
      const bool useTopDownBitmap = bitmapThreshold > 0.0 && frontierFraction < bitmapThreshold;

      if constexpr( std::is_same_v< DeviceType, Devices::Host > ) {
         auto hostAtomicYView = hostAtomicY.getView();

         if( useTopDownBitmap ) {
            TNL::Algorithms::parallelFor< DeviceType >(
               0,
               n,
               [ = ] __cuda_callable__( IndexType idx ) mutable
               {
                  hostAtomicYView[ idx ] = yView[ idx ];
               } );

            auto marksView_bitmap = marks.getView();
            auto nextMarksView = nextMarks.getView();

            forAllEdges(
               graph,
               [ = ] __cuda_callable__(
                  IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
               {
                  if( marksView_bitmap[ sourceIdx ] != 1 )
                     return;
                  if( targetIdx == Matrices::paddingIndex< IndexType > )
                     return;

                  const ValueType transformedWeight = edgeWeightCallable( sourceIdx, targetIdx, weight );
                  if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
                     return;

                  const ValueType newDistance = yView[ sourceIdx ] + transformedWeight;
                  const ValueType oldDistance = hostAtomicYView[ targetIdx ].fetch_min( newDistance );
                  if( newDistance < oldDistance ) {
#if defined( HAVE_OPENMP )
                  #pragma omp atomic write
#endif
                     predecessorsView[ targetIdx ] = sourceIdx;
                     nextMarksView[ targetIdx ] = 1;
                  }
               },
               launchConfig );

            TNL::Algorithms::parallelFor< DeviceType >(
               0,
               n,
               [ = ] __cuda_callable__( IndexType idx ) mutable
               {
                  yView[ idx ] = hostAtomicYView[ idx ].load();
               } );

            frontierSize = sum( nextMarks );
            marks.swap( nextMarks );
            nextMarks = 0;
         }
         else {
            marks = 0;

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
                  if( targetIdx != Matrices::paddingIndex< IndexType > ) {
                     const ValueType transformedWeight = edgeWeightCallable( sourceIdx, targetIdx, weight );
                     if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
                        return;

                     ValueType newDistance = yView[ sourceIdx ] + transformedWeight;
                     const ValueType oldDistance = hostAtomicYView[ targetIdx ].fetch_min( newDistance );
                     if( newDistance < oldDistance ) {
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

            TNL::Algorithms::parallelFor< DeviceType >(
               0,
               n,
               [ = ] __cuda_callable__( IndexType idx ) mutable
               {
                  yView[ idx ] = hostAtomicYView[ idx ].load();
               } );

            frontierSize = detail::compactFrontier< DeviceType, IndexType >( marks, marksScan, frontier );
         }
      }
      else {  // DeviceType != Host
         if( useTopDownBitmap ) {
            auto marksView_bitmap = marks.getView();
            auto nextMarksView = nextMarks.getView();

            forAllEdges(
               graph,
               [ = ] __cuda_callable__(
                  IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
               {
                  TNL_ASSERT_GE( sourceIdx, 0, "" );
                  TNL_ASSERT_LT( sourceIdx, yView.getSize(), "" );
                  TNL_ASSERT_GE( targetIdx, 0, "" );
                  TNL_ASSERT_LT( targetIdx, yView.getSize(), "" );

                  if( marksView_bitmap[ sourceIdx ] != 1 )
                     return;
                  if( targetIdx == Matrices::paddingIndex< IndexType > )
                     return;

                  const ValueType transformedWeight = edgeWeightCallable( sourceIdx, targetIdx, weight );
                  if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
                     return;

                  const ValueType newDistance = yView[ sourceIdx ] + transformedWeight;
                  if( newDistance < yView[ targetIdx ] ) {
                     atomicMin( &yView[ targetIdx ], newDistance );
                     atomicMin( &predecessorsView[ targetIdx ], sourceIdx );
                     nextMarksView[ targetIdx ] = 1;
                  }
               },
               launchConfig );

            frontierSize = sum( nextMarks );
            marks.swap( nextMarks );
            nextMarks = 0;
         }
         else {
            marks = 0;

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
                  if( targetIdx != Matrices::paddingIndex< IndexType > ) {
                     const ValueType transformedWeight = edgeWeightCallable( sourceIdx, targetIdx, weight );
                     if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
                        return;

                     ValueType newDistance = yView[ sourceIdx ] + transformedWeight;
                     if( newDistance < yView[ targetIdx ] ) {
                        atomicMin( &yView[ targetIdx ], newDistance );
                        atomicMin( &predecessorsView[ targetIdx ], sourceIdx );
                        atomicMax( &marksView[ targetIdx ], 1 );
                     }
                  }
               },
               launchConfig );

            frontierSize = detail::compactFrontier< DeviceType, IndexType >( marks, marksScan, frontier );
         }
      }

      if( frontierSize == 0 )
         break;
      distances = y;
   }
}

template< typename Graph, typename Vector, typename EdgeWeightCallable, typename Index >
void
singleSourceShortestPath_impl(
   const Graph& graph,
   Index start,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   double bitmapThreshold,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert(
      ! Graph::AdjacencyMatrixType::MatrixType::isSymmetric(), "SSSP requires general adjacency matrix, not symmetric." );

   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   const auto graphView = graph.getConstView();

   distances.setSize( graph.getVertexCount() );
   if( graph.getVertexCount() == 0 )
      return;
   TNL_ASSERT_GE( start, static_cast< Index >( 0 ), "Start vertex index must be non-negative." );
   TNL_ASSERT_LT( start, graph.getVertexCount(), "Start vertex index must be less than the number of vertices." );

   // Use 5-arg reduce (explicit Result+identity) to avoid decltype(fetch(0)):
   // NVCC may evaluate it on host when fetch captures a GPU view via nested
   // extended-lambda forwarding.
   const bool startActive = TNL::Algorithms::reduce< DeviceType, Index, bool >(
      0,
      1,
      [ = ] __cuda_callable__( Index ) -> bool
      {
         return graphView.vertexExists( start );
      },
      TNL::LogicalAnd{},
      true );
   if( ! startActive )
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
            if( ! graphView.vertexExists( neighbor ) )
               continue;
            if( ! graphView.edgeExists( current, neighbor, edgeWeight ) )
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
      parallelSingleSourceShortestPath( graph, start, edgeWeightCallable, distances, bitmapThreshold, launchConfig );
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
   double bitmapThreshold,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using ValueType = typename Graph::ValueType;
   singleSourceShortestPath_impl(
      graph,
      start,
      [] __cuda_callable__( Index, Index, ValueType weight )
      {
         return weight;
      },
      distances,
      bitmapThreshold,
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgeWeightCallable, typename Index, typename Enable >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   double bitmapThreshold,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert(
      detail::isEdgeWeightCallable_v< EdgeWeightCallable, Graph >,
      "SSSP edge-weight callable must return ValueType and accept (source, target, weight)." );
   singleSourceShortestPath_impl(
      graph, start, std::forward< EdgeWeightCallable >( edgeWeightCallable ), distances, bitmapThreshold, launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
