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
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Algorithms/contains.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

#include "details/activeVertices.hpp"
#include "details/lambdaTraits.hpp"
#include "details/parallelTraversal.hpp"
#include "breadthFirstSearch.h"

namespace TNL::Graphs::Algorithms {

template< typename Graph, typename Visitor, typename Vector >
void
breadthFirstSearchParallel(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   const TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   const IndexType n = graph.getVertexCount();
   distances.setSize( n );

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
   for( IndexType i = 0; i < n; i++ ) {
      marks = 0;
      if constexpr( std::is_same_v< DeviceType, Devices::Host > ) {
         forEdges(
            graph,
            frontier,
            0,
            frontierSize,
            [ = ] __cuda_callable__(
               IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
            {
               // NOTE: Reading yView[targetIdx] without synchronization while another
               // thread may write to it is technically a data race. In practice all
               // concurrent writers in the same layer write the same value (i+1), so
               // the result is correct, but this is undefined behavior per the C++ standard.
               if( targetIdx != Matrices::paddingIndex< IndexType > && yView[ targetIdx ] == -1 ) {
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
      }
      else {
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
               // edgeExists and isActive(target) are applied by the SubGraph forEdges wrapper.
               if( targetIdx != Matrices::paddingIndex< IndexType > && yView[ targetIdx ] == -1 ) {
                  atomicMax( &yView[ targetIdx ], i + 1 );
                  atomicMin( &predecessorsView[ targetIdx ], sourceIdx );
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

template< typename Graph, typename Visitor, typename Vector >
void
breadthFirstSearch_impl(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   const TNL::Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   static_assert(
      ! Graph::AdjacencyMatrixType::MatrixType::isSymmetric(), "BFS requires general adjacency matrix, not symmetric." );
   using IndexType = typename Graph::IndexType;
   using DeviceType = typename Graph::DeviceType;
   const auto& adjacencyMatrix = graph.getAdjacencyMatrix();
   const IndexType n = graph.getVertexCount();
   const auto graphView = graph.getConstView();

   distances.setSize( n );
   if( n == 0 )
      return;
   TNL_ASSERT_GE( start, static_cast< IndexType >( 0 ), "Start vertex index must be non-negative." );
   TNL_ASSERT_LT( start, n, "Start vertex index must be less than the number of vertices." );

   // Use 5-arg reduce (explicit Result+identity) to avoid decltype(fetch(0)):
   // NVCC may evaluate it on host when fetch captures a GPU view via nested
   // extended-lambda forwarding (SCC → breadthFirstSearchIf → _impl).
   const bool startActive = TNL::Algorithms::reduce< DeviceType, IndexType, bool >(
      0,
      1,
      [ = ] __cuda_callable__( IndexType ) -> bool
      {
         return graphView.isActive( start );
      },
      TNL::LogicalAnd{},
      true );
   if( ! startActive )
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
            if( ! graphView.isActive( neighbor ) )
               continue;
            if( ! graphView.edgeExists( current, neighbor, edgeWeight ) )
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
      breadthFirstSearchParallel( graph, start, visitor, distances, launchConfig );
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
   breadthFirstSearch_impl(
      graph, start, [] __cuda_callable__( typename Graph::IndexType, typename Graph::IndexType ) {}, distances, launchConfig );
}

template< typename Graph, typename Vector, typename Visitor, typename Enable >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( detail::isBfsVisitor_v< Visitor, Graph >, "BFS visitor must accept (node, distance)." );
   breadthFirstSearch_impl( graph, start, std::forward< Visitor >( visitor ), distances, launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
