// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>
#include <stdexcept>
#include <type_traits>

#include <TNL/Devices/Sequential.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Functional.h>
#include <TNL/Assert.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Algorithms/contains.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

#include "details/activeVertices.hpp"
#include "details/parallelTraversal.hpp"
#include "breadthFirstSearch.h"

namespace TNL::Graphs::Algorithms {

namespace detail {

template< typename EdgePredicate, typename Graph >
struct IsBfsEdgePredicate
: std::bool_constant<
     std::
        is_invocable_r_v< bool, EdgePredicate, typename Graph::IndexType, typename Graph::IndexType, typename Graph::ValueType > >
{};

template< typename EdgePredicate, typename Graph >
constexpr bool isBfsEdgePredicate_v = IsBfsEdgePredicate< EdgePredicate, Graph >::value;

}  // namespace detail

template< typename Graph, typename Visitor, typename ActivePredicate, typename EdgePredicate, typename Vector >
void
breadthFirstSearchParallel(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   ActivePredicate&& isActive,
   EdgePredicate&& edgePredicate,
   Vector& distances,
   const TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   const IndexType n = graph.getVertexCount();
   distances.setSize( n );

   // Level-synchronous BFS: each iteration processes one frontier (all vertices
   // at the same distance from the source) and builds the next frontier.
   //
   // y           – working copy of distances (updated by concurrent threads)
   // predecessors – parent vertex for each visited node
   // marks       – 1 if the vertex was newly discovered in this iteration, 0 otherwise
   // marksScan   – inclusive prefix sum of marks (used to compact the next frontier)
   // frontier    – dense array of vertex indices forming the current frontier
   Vector y( distances.getSize() );
   Containers::Vector< IndexType, DeviceType, IndexType > predecessors( n, -1 );
   Containers::Vector< IndexType, DeviceType, IndexType > marks( n );
   Containers::Vector< IndexType, DeviceType, IndexType > marksScan( n, 0 );
   Containers::Vector< IndexType, DeviceType, IndexType > frontier( n, 0 );
   distances = -1;
   distances.setElement( start, 0 );
   frontier.setElement( 0, start );
   IndexType frontierSize( 1 );
   y = distances;
   auto yView = y.getView();
   auto predecessorsView = predecessors.getView();
   auto marksView = marks.getView();
   for( IndexType i = 0; i <= n; i++ ) {
      marks = 0;
      if constexpr( std::is_same_v< DeviceType, Devices::Host > )
         forEdges(
            graph,
            frontier,
            0,
            frontierSize,
            [ = ] __cuda_callable__( IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
            {
               if( targetIdx != Matrices::paddingIndex< IndexType > && isActive( targetIdx ) && yView[ targetIdx ] == -1
                   && edgePredicate( sourceIdx, targetIdx, weight ) )
               {
#if defined( HAVE_OPENMP )
   #pragma omp atomic write
#endif
                  yView[ targetIdx ] = i + 1;
#if defined( HAVE_OPENMP )
   #pragma omp atomic write
#endif
                  predecessorsView[ targetIdx ] = sourceIdx;
#if defined( HAVE_OPENMP )
   #pragma omp atomic write
#endif
                  marksView[ targetIdx ] = 1;
                  visitor( targetIdx, i + 1 );
               }
            },
            launchConfig );
      else {
         forEdges(
            graph,
            frontier,
            0,
            frontierSize,
            [ = ] __cuda_callable__( IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
            {
               TNL_ASSERT_GE( sourceIdx, 0, "" );
               TNL_ASSERT_LT( sourceIdx, yView.getSize(), "" );
               TNL_ASSERT_GE( targetIdx, 0, "" );
               TNL_ASSERT_LT( targetIdx, yView.getSize(), "" );
               if( targetIdx != Matrices::paddingIndex< IndexType > && isActive( targetIdx ) && yView[ targetIdx ] == -1
                   && edgePredicate( sourceIdx, targetIdx, weight ) )
               {
                  // atomicMax is safe for distances: i+1 is always >= -1 (the
                  // initial sentinel), so the first writer wins and concurrent
                  // writers in the same layer write the same value.
                  // The predecessor, however, is set to the *largest* source
                  // index among concurrent discoverers, not necessarily the
                  // first one.  This is acceptable for BFS (all sources are in
                  // the same layer), but makes the result non-deterministic
                  // with respect to the sequential version.
                  atomicMax( &yView[ targetIdx ], i + 1 );
                  atomicMax( &predecessorsView[ targetIdx ], sourceIdx );
                  atomicMax( &marksView[ targetIdx ], 1 );
                  visitor( targetIdx, i + 1 );
               }
            },
            launchConfig );
      }
       // Compact newly discovered vertices into the next frontier
      frontierSize = detail::compactFrontier< DeviceType, IndexType >( marks, marksScan, frontier );
      if( frontierSize == 0 )
         break;

      distances = y;
   }
}

template< typename Graph, typename Visitor, typename ActivePredicate, typename EdgePredicate, typename Vector >
void
breadthFirstSearch_impl(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   ActivePredicate&& isActive,
   EdgePredicate&& edgePredicate,
   Vector& distances,
   const TNL::Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   static_assert(
      ! Graph::AdjacencyMatrixType::MatrixType::isSymmetric(), "BFS requires general adjacency matrix, not symmetric." );
   using IndexType = typename Graph::IndexType;
   using DeviceType = typename Graph::DeviceType;
   const auto& adjacencyMatrix = graph.getAdjacencyMatrix();
   const IndexType n = graph.getVertexCount();

   distances.setSize( n );
   if( n == 0 )
      return;
   TNL_ASSERT_GE( start, static_cast< IndexType >( 0 ), "Start vertex index must be non-negative." );
   TNL_ASSERT_LT( start, n, "Start vertex index must be less than the number of vertices." );

   if( ! isActive( start ) )
      throw std::invalid_argument( "Start vertex must belong to the induced active subgraph." );

   if constexpr( std::is_same_v< DeviceType, TNL::Devices::Sequential > ) {
      distances = -1;
      distances.setElement( start, 0.0 );

      std::queue< IndexType > q;
      q.push( start );

      while( ! q.empty() ) {
         IndexType current = q.front();
         q.pop();

         const auto row = adjacencyMatrix.getRow( current );
         for( IndexType i = 0; i < row.getSize(); i++ ) {
            const auto& edgeWeight = row.getValue( i );
            const auto& neighbor = row.getColumnIndex( i );
            if( neighbor == Matrices::paddingIndex< IndexType > )
               continue;
            if( ! isActive( neighbor ) )
               continue;
            if( ! edgePredicate( current, neighbor, edgeWeight ) )
               continue;

            if( distances[ neighbor ] == -1 ) {
               IndexType distance = distances[ current ] + 1;
               distances[ neighbor ] = distance;
               visitor( neighbor, distance );
               q.push( neighbor );
            }
         }
      }
   }
   else {
      breadthFirstSearchParallel( graph, start, visitor, isActive, edgePredicate, distances, launchConfig );
   }
}

template< typename Graph, typename Vector >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   breadthFirstSearch_impl(
      graph,
      start,
      [] __cuda_callable__( IndexType, IndexType ) {},
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgePredicate, typename >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   EdgePredicate&& edgePredicate,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   static_assert(
      detail::isBfsEdgePredicate_v< EdgePredicate, Graph >,
      "BFS edge predicate must return bool and accept (source, target) or (source, target, weight)." );

   breadthFirstSearch_impl(
      graph,
      start,
      [] __cuda_callable__( IndexType, IndexType ) {},
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      std::forward< EdgePredicate >( edgePredicate ),
      distances,
      launchConfig );
}

template< typename Graph, typename Vector, typename Visitor, typename >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   breadthFirstSearch_impl(
      graph,
      start,
      visitor,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   const auto isActive = [ = ] __cuda_callable__( IndexType vertex )
   {
      return static_cast< bool >( activeVerticesView[ vertex ] );
   };
   breadthFirstSearch_impl(
      graph,
      start,
      [] __cuda_callable__( IndexType, IndexType ) {},
      isActive,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename EdgePredicate, typename >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   static_assert(
      detail::isBfsEdgePredicate_v< EdgePredicate, Graph >,
      "BFS edge predicate must return bool and accept (source, target) or (source, target, weight)." );

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   const auto isActive = [ = ] __cuda_callable__( IndexType vertex )
   {
      return static_cast< bool >( activeVerticesView[ vertex ] );
   };
   breadthFirstSearch_impl(
      graph,
      start,
      [] __cuda_callable__( IndexType, IndexType ) {},
      isActive,
      std::forward< EdgePredicate >( edgePredicate ),
      distances,
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector >
void
breadthFirstSearchIf(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   breadthFirstSearch_impl(
      graph,
      start,
      [] __cuda_callable__( IndexType, IndexType ) {},
      predicate,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
breadthFirstSearchIf(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   static_assert(
      detail::isBfsEdgePredicate_v< EdgePredicate, Graph >,
      "BFS edge predicate must return bool and accept (source, target) or (source, target, weight)." );

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   breadthFirstSearch_impl(
      graph,
      start,
      [] __cuda_callable__( IndexType, IndexType ) {},
      predicate,
      std::forward< EdgePredicate >( edgePredicate ),
      distances,
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename Visitor, typename >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   const auto isActive = [ = ] __cuda_callable__( IndexType vertex )
   {
      return static_cast< bool >( activeVerticesView[ vertex ] );
   };
   breadthFirstSearch_impl(
      graph,
      start,
      visitor,
      isActive,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      distances,
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename Visitor >
void
breadthFirstSearchIfWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   breadthFirstSearch_impl(
      graph,
      start,
      visitor,
      predicate,
      [] __cuda_callable__( IndexType, IndexType, const ValueType& )
      {
         return true;
      },
      distances,
      launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
