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
   return weight == std::numeric_limits< Real >::infinity() || weight == -std::numeric_limits< Real >::infinity();
}

}  // namespace detail

template<
   typename Graph,
   typename Vector,
   typename ActivePredicate,
   typename EdgeWeightCallable,
   typename Index = typename Graph::IndexType >
void
parallelSingleSourceShortestPath(
   const Graph& graph,
   Index start,
   ActivePredicate&& isActive,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using Real = typename Graph::ValueType;
   using Device = typename Graph::DeviceType;
   const Index n = graph.getVertexCount();
   distances.setSize( n );

   Vector y( distances.getSize() );
   Containers::Vector< Index, Device, Index > predecessors( n, -1 );
   Containers::Vector< Index, Device, Index > marks( n );
   Containers::Vector< Index, Device, Index > marks_scan( n, 0 );
   Containers::Vector< Index, Device, Index > frontier( n, 0 );
   distances = std::numeric_limits< Real >::max();
   distances.setElement( start, 0 );
   frontier.setElement( 0, start );
   Index frontier_size( 1 );
   y = distances;
   auto y_view = y.getView();
   auto predecessors_view = predecessors.getView();
   auto marks_view = marks.getView();
   auto marks_scan_view = marks_scan.getView();
   for( Index i = 0; i <= n; i++ ) {
      marks = 0;
      if constexpr( std::is_same_v< Device, Devices::Host > )
         forEdges(
            graph,
            frontier,
            0,
            frontier_size,
            [ = ] __cuda_callable__( Index sourceIdx, Index localIdx, Index targetIdx, const Real& weight ) mutable
            {
               if( targetIdx != Matrices::paddingIndex< Index > && isActive( targetIdx ) ) {
                  const Real transformedWeight = edgeWeightCallable( sourceIdx, targetIdx, weight );
                  if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
                     return;

                  Real new_distance = y_view[ sourceIdx ] + transformedWeight;
                  if( new_distance < y_view[ targetIdx ] ) {
#if defined( HAVE_OPENMP )
   #pragma omp atomic write
#endif
                     y_view[ targetIdx ] = new_distance;
#if defined( HAVE_OPENMP )
   #pragma omp atomic write
#endif
                     predecessors_view[ targetIdx ] = sourceIdx;
#if defined( HAVE_OPENMP )
   #pragma omp atomic write
#endif
                     marks_view[ targetIdx ] = 1;
                  }
               }
            },
            launchConfig );
      else
         forEdges(
            graph,
            frontier,
            0,
            frontier_size,
            [ = ] __cuda_callable__( Index sourceIdx, Index localIdx, Index targetIdx, const Real& weight ) mutable
            {
               TNL_ASSERT_GE( sourceIdx, 0, "" );
               TNL_ASSERT_LT( sourceIdx, y_view.getSize(), "" );
               TNL_ASSERT_GE( targetIdx, 0, "" );
               TNL_ASSERT_LT( targetIdx, y_view.getSize(), "" );
               if( targetIdx != Matrices::paddingIndex< Index > && isActive( targetIdx ) ) {
                  const Real transformedWeight = edgeWeightCallable( sourceIdx, targetIdx, weight );
                  if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
                     return;

                  Real new_distance = y_view[ sourceIdx ] + transformedWeight;
                  if( new_distance < y_view[ targetIdx ] ) {
                     atomicMin( &y_view[ targetIdx ], new_distance );
                     atomicMin( &predecessors_view[ targetIdx ], sourceIdx );
                     atomicMax( &marks_view[ targetIdx ], 1 );
                  }
               }
            },
            launchConfig );
      TNL::Algorithms::inclusiveScan( marks, marks_scan );
      frontier_size = marks_scan.getElement( n - 1 );
      if( frontier_size == 0 )
         break;
      frontier = 0;
      auto frontier_view = frontier.getView();
      auto f = [ = ] __cuda_callable__( const Index idx, const Index value ) mutable
      {
         if( idx == 0 ) {
            if( marks_scan_view[ 0 ] == 1 )
               frontier_view[ 0 ] = idx;
         }
         else if( marks_scan_view[ idx ] - marks_scan_view[ idx - 1 ] == 1 )
            frontier_view[ marks_scan_view[ idx ] - 1 ] = idx;
      };
      marks_scan.forAllElements( f );
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

   using Real = typename Graph::ValueType;
   using Device = typename Graph::DeviceType;

   distances.setSize( graph.getVertexCount() );
   if( graph.getVertexCount() == 0 )
      return;
   TNL_ASSERT_GE( start, static_cast< Index >( 0 ), "Start vertex index must be non-negative." );
   TNL_ASSERT_LT( start, graph.getVertexCount(), "Start vertex index must be less than the number of vertices." );

   if( ! isActive( start ) )
      throw std::invalid_argument( "Start vertex must belong to the induced active subgraph." );

   distances = std::numeric_limits< Real >::max();
   distances.setElement( start, 0.0 );

   // In the sequential version, we use the Dijkstra algorithm.
   if constexpr( std::is_same_v< Device, TNL::Devices::Sequential > ) {
      // The priority queue stores pairs of (distance, vertex)
      std::priority_queue< std::pair< Real, Index >, std::vector< std::pair< Real, Index > >, std::greater<> > pq;
      pq.emplace( 0, start );

      while( ! pq.empty() ) {
         Real current_distance;
         Index current;
         std::tie( current_distance, current ) = pq.top();
         pq.pop();

         if( current_distance > distances[ current ] ) {
            continue;
         }

         const auto row = graph.getAdjacencyMatrix().getRow( current );
         for( Index i = 0; i < row.getSize(); i++ ) {
            const auto& edge_weight = row.getValue( i );
            const auto& neighbor = row.getColumnIndex( i );
            if( neighbor == Matrices::paddingIndex< Index > )
               continue;
            if( ! isActive( neighbor ) )
               continue;

            const Real transformedWeight = edgeWeightCallable( current, neighbor, edge_weight );
            if( detail::isBlockedSsspEdgeWeight( transformedWeight ) )
               continue;

            const Real distance = current_distance + transformedWeight;

            if( distance < distances[ neighbor ] ) {
               distances[ neighbor ] = distance;
               pq.emplace( distance, neighbor );
            }
         }
      }
   }
   else {
      parallelSingleSourceShortestPath( graph, start, isActive, edgeWeightCallable, distances, launchConfig );
   }
   distances.forAllElements(
      [] __cuda_callable__( Index i, Real & x )
      {
         x = ( x == std::numeric_limits< Real >::max() ) ? -1.0 : x;
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
      [] __cuda_callable__( Index, Index, auto weight )
      {
         return weight;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgeWeightCallable, typename Index, typename >
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

template< typename Graph, typename VertexIndexes, typename Vector, typename Index, typename >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   const VertexIndexes& vertexIndexes,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using Device = typename Graph::DeviceType;
   using IndexVector = Containers::Vector< Index, Device, Index >;

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
      [] __cuda_callable__( Index, Index, auto weight )
      {
         return weight;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename EdgeWeightCallable, typename Index, typename >
void
singleSourceShortestPath(
   const Graph& graph,
   Index start,
   const VertexIndexes& vertexIndexes,
   EdgeWeightCallable&& edgeWeightCallable,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using Device = typename Graph::DeviceType;
   using IndexVector = Containers::Vector< Index, Device, Index >;

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
      [] __cuda_callable__( Index, Index, auto weight )
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
