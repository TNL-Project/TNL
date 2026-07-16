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
#include <TNL/Atomic.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Algorithms/contains.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

#include "details/activeVertices.hpp"
#include "details/lambdaTraits.hpp"
#include "details/parallelTraversal.hpp"
#include "breadthFirstSearch.h"

namespace TNL::Graphs::Algorithms {

template< bool WithPredecessors, typename Graph, typename Visitor, typename Vector, typename PredecessorVector >
void
breadthFirstSearchParallel(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   PredecessorVector& predecessors,
   bool deterministic,
   const TNL::Algorithms::Segments::LaunchConfiguration launchConfig,
   double bypassThreshold )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   const IndexType n = graph.getVertexCount();
   distances.setSize( n );
   if( n == 0 )
      return;

   distances = -1;
   distances.setElement( start, 0 );

   if constexpr( WithPredecessors ) {
      predecessors.setSize( n );
      if( deterministic )
         predecessors = n;
      else
         predecessors = -1;
      predecessors.setElement( start, -1 );
   }

   // marks: current frontier bitmap (input for bypass, output for compact)
   // nextMarks: next frontier bitmap (output for bypass)
   Containers::Vector< IndexType, DeviceType, IndexType > marks( n );
   Containers::Vector< IndexType, DeviceType, IndexType > marksScan( n, 0 );
   Containers::Vector< IndexType, DeviceType, IndexType > frontier( n, 0 );
   Containers::Vector< IndexType, DeviceType, IndexType > nextMarks( n, 0 );
   frontier.setElement( 0, start );
   IndexType frontierSize( 1 );

   // Initialize marks as frontier bitmap for potential bypass on i=0
   marks = 0;
   marks.setElement( start, 1 );

   auto predecessorsView = predecessors.getView();

   if constexpr( std::is_same_v< DeviceType, Devices::Host > ) {
      using HostAtomicIntVec = Containers::Vector< Atomic< IndexType, Devices::Host >, Devices::Host, IndexType >;
      HostAtomicIntVec atomicDistances( n );
      auto atomicDistancesView = atomicDistances.getView();
      auto distancesView = distances.getView();

      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         n,
         [ = ] __cuda_callable__( IndexType idx ) mutable
         {
            atomicDistancesView[ idx ] = distancesView[ idx ];
         } );

      HostAtomicIntVec atomicPredecessors( n );
      auto atomicPredView = atomicPredecessors.getView();
      if constexpr( WithPredecessors ) {
         if( deterministic ) {
            TNL::Algorithms::parallelFor< DeviceType >(
               0,
               n,
               [ = ] __cuda_callable__( IndexType idx ) mutable
               {
                  atomicPredView[ idx ] = predecessorsView[ idx ];
               } );
         }
      }

      bool previousWasBypass = false;
      for( IndexType i = 0; i < n; i++ ) {
         // When the frontier is small, skip the O(n) compactFrontier (prefix
         // scan + scatter) and instead scan all edges with a cheap marks check.
         const bool useBypass =
            bypassThreshold > 0.0 && static_cast< double >( frontierSize ) / static_cast< double >( n ) < bypassThreshold;

         if( useBypass ) {
            auto marksView_bypass = marks.getView();
            auto nextMarksView = nextMarks.getView();

            forAllEdges(
               graph,
               [ = ] __cuda_callable__(
                  IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
               {
                  // NVCC forbids first-capture of variables inside if constexpr
                  // in extended lambdas.  These (void) casts force early capture.
                  (void) deterministic;
                  (void) predecessorsView;
                  (void) atomicPredView;

                  if( marksView_bypass[ sourceIdx ] != 1 )
                     return;

                  if( targetIdx == Matrices::paddingIndex< IndexType > )
                     return;

                  IndexType prev = -1;
                  const bool i_am_winner = atomicDistancesView[ targetIdx ].compare_exchange_strong( prev, i + 1 );

                  if( i_am_winner ) {
                     nextMarksView[ targetIdx ] = 1;
                     visitor( targetIdx, i + 1 );
                  }

                  // Deterministic mode: all threads that observe the vertex at
                  // distance i+1 compete via fetch_min, so the smallest source
                  // index wins.  This must run even for non-winners (prev == i+1),
                  // hence it cannot be nested inside the i_am_winner block above.
                  if constexpr( WithPredecessors ) {
                     if( deterministic ) {
                        if( i_am_winner || prev == i + 1 )
                           atomicPredView[ targetIdx ].fetch_min( sourceIdx );
                     }
                     else if( i_am_winner ) {
                        predecessorsView[ targetIdx ] = sourceIdx;
                     }
                  }
               },
               launchConfig );

            frontierSize = sum( nextMarks );

            marks.swap( nextMarks );
            nextMarks = 0;
            previousWasBypass = true;
         }
         else {
            if( previousWasBypass ) {
               frontierSize = detail::compactFrontier< DeviceType, IndexType >( marks, marksScan, frontier );
               previousWasBypass = false;
            }

            marks = 0;
            auto marksView = marks.getView();

            forEdges(
               graph,
               frontier,
               0,
               frontierSize,
               [ = ] __cuda_callable__(
                  IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
               {
                  // NVCC forbids first-capture of variables inside if constexpr
                  // in extended lambdas.  These (void) casts force early capture.
                  (void) deterministic;
                  (void) predecessorsView;
                  (void) atomicPredView;

                  if( targetIdx == Matrices::paddingIndex< IndexType > )
                     return;

                  IndexType prev = -1;
                  const bool i_am_winner = atomicDistancesView[ targetIdx ].compare_exchange_strong( prev, i + 1 );

                  if( i_am_winner ) {
                     marksView[ targetIdx ] = 1;
                     visitor( targetIdx, i + 1 );
                  }

                  // Deterministic mode: all threads that observe the vertex at
                  // distance i+1 compete via fetch_min, so the smallest source
                  // index wins.  This must run even for non-winners (prev == i+1),
                  // hence it cannot be nested inside the i_am_winner block above.
                  if constexpr( WithPredecessors ) {
                     if( deterministic ) {
                        if( i_am_winner || prev == i + 1 )
                           atomicPredView[ targetIdx ].fetch_min( sourceIdx );
                     }
                     else if( i_am_winner ) {
                        predecessorsView[ targetIdx ] = sourceIdx;
                     }
                  }
               },
               launchConfig );

            frontierSize = detail::compactFrontier< DeviceType, IndexType >( marks, marksScan, frontier );
         }

         if( frontierSize == 0 )
            break;
      }

      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         n,
         [ = ] __cuda_callable__( IndexType idx ) mutable
         {
            distancesView[ idx ] = atomicDistancesView[ idx ].load();
         } );

      if constexpr( WithPredecessors ) {
         if( deterministic ) {
            TNL::Algorithms::parallelFor< DeviceType >(
               0,
               n,
               [ = ] __cuda_callable__( IndexType idx ) mutable
               {
                  IndexType p = atomicPredView[ idx ].load();
                  predecessorsView[ idx ] = ( p == n ) ? -1 : p;
               } );
         }
      }
   }
   else {
      auto distancesView = distances.getView();

      bool previousWasBypass = false;
      for( IndexType i = 0; i < n; i++ ) {
         // When the frontier is small, skip the O(n) compactFrontier (prefix
         // scan + scatter) and instead scan all edges with a cheap marks check.
         const bool useBypass =
            bypassThreshold > 0.0 && static_cast< double >( frontierSize ) / static_cast< double >( n ) < bypassThreshold;

         if( useBypass ) {
            auto marksView_bypass = marks.getView();
            auto nextMarksView = nextMarks.getView();

            forAllEdges(
               graph,
               [ = ] __cuda_callable__(
                  IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
               {
                  // NVCC forbids first-capture of variables inside if constexpr
                  // in extended lambdas.  These (void) casts force early capture.
                  (void) deterministic;
                  (void) predecessorsView;

                  TNL_ASSERT_GE( sourceIdx, 0, "" );
                  TNL_ASSERT_LT( sourceIdx, distancesView.getSize(), "" );
                  TNL_ASSERT_GE( targetIdx, 0, "" );
                  TNL_ASSERT_LT( targetIdx, distancesView.getSize(), "" );

                  if( marksView_bypass[ sourceIdx ] != 1 )
                     return;

                  if( targetIdx == Matrices::paddingIndex< IndexType > )
                     return;

                  const IndexType old = atomicCAS( &distancesView[ targetIdx ], -1, i + 1 );
                  const bool i_am_winner = ( old == -1 );

                  if( i_am_winner ) {
                     nextMarksView[ targetIdx ] = 1;
                     visitor( targetIdx, i + 1 );
                  }

                  // Deterministic mode: all threads that observe the vertex at
                  // distance i+1 compete via atomicMin, so the smallest source
                  // index wins.  This must run even for non-winners (old == i+1),
                  // hence it cannot be nested inside the i_am_winner block above.
                  if constexpr( WithPredecessors ) {
                     if( deterministic ) {
                        if( i_am_winner || old == i + 1 )
                           atomicMin( &predecessorsView[ targetIdx ], sourceIdx );
                     }
                     else if( i_am_winner ) {
                        predecessorsView[ targetIdx ] = sourceIdx;
                     }
                  }
               },
               launchConfig );

            frontierSize = sum( nextMarks );

            marks.swap( nextMarks );
            nextMarks = 0;
            previousWasBypass = true;
         }
         else {
            if( previousWasBypass ) {
               frontierSize = detail::compactFrontier< DeviceType, IndexType >( marks, marksScan, frontier );
               previousWasBypass = false;
            }

            marks = 0;
            auto marksView = marks.getView();

            forEdges(
               graph,
               frontier,
               0,
               frontierSize,
               [ = ] __cuda_callable__(
                  IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
               {
                  // NVCC forbids first-capture of variables inside if constexpr
                  // in extended lambdas.  These (void) casts force early capture.
                  (void) deterministic;
                  (void) predecessorsView;

                  TNL_ASSERT_GE( sourceIdx, 0, "" );
                  TNL_ASSERT_LT( sourceIdx, distancesView.getSize(), "" );
                  TNL_ASSERT_GE( targetIdx, 0, "" );
                  TNL_ASSERT_LT( targetIdx, distancesView.getSize(), "" );

                  if( targetIdx == Matrices::paddingIndex< IndexType > )
                     return;

                  const IndexType old = atomicCAS( &distancesView[ targetIdx ], -1, i + 1 );
                  const bool i_am_winner = ( old == -1 );

                  if( i_am_winner ) {
                     marksView[ targetIdx ] = 1;
                     visitor( targetIdx, i + 1 );
                  }

                  // Deterministic mode: all threads that observe the vertex at
                  // distance i+1 compete via atomicMin, so the smallest source
                  // index wins.  This must run even for non-winners (old == i+1),
                  // hence it cannot be nested inside the i_am_winner block above.
                  if constexpr( WithPredecessors ) {
                     if( deterministic ) {
                        if( i_am_winner || old == i + 1 )
                           atomicMin( &predecessorsView[ targetIdx ], sourceIdx );
                     }
                     else if( i_am_winner ) {
                        predecessorsView[ targetIdx ] = sourceIdx;
                     }
                  }
               },
               launchConfig );

            frontierSize = detail::compactFrontier< DeviceType, IndexType >( marks, marksScan, frontier );
         }

         if( frontierSize == 0 )
            break;
      }

      if constexpr( WithPredecessors ) {
         if( deterministic ) {
            TNL::Algorithms::parallelFor< DeviceType >(
               0,
               n,
               [ = ] __cuda_callable__( IndexType idx ) mutable
               {
                  if( predecessorsView[ idx ] == n )
                     predecessorsView[ idx ] = -1;
               } );
         }
      }
   }
}

template< bool WithPredecessors, typename Graph, typename Visitor, typename Vector, typename PredecessorVector >
void
breadthFirstSearch_impl(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   PredecessorVector& predecessors,
   bool deterministic,
   const TNL::Algorithms::Segments::LaunchConfiguration& launchConfig,
   double bypassThreshold )
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

   const bool startActive = TNL::Algorithms::reduce< DeviceType, IndexType, bool >(
      0,
      1,
      [ = ] __cuda_callable__( IndexType ) -> bool
      {
         return graphView.vertexExists( start );
      },
      TNL::LogicalAnd{},
      true );
   if( ! startActive )
      throw std::invalid_argument( "Start vertex must belong to the induced active subgraph." );

   if constexpr( std::is_same_v< DeviceType, TNL::Devices::Sequential > ) {
      distances = -1;
      distances.setElement( start, 0 );

      if constexpr( WithPredecessors ) {
         predecessors.setSize( n );
         predecessors = -1;
      }

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
            if( ! graphView.vertexExists( neighbor ) )
               continue;
            if( ! graphView.edgeExists( current, neighbor, edgeWeight ) )
               continue;

            if( distances[ neighbor ] == -1 ) {
               IndexType distance = distances[ current ] + 1;
               distances[ neighbor ] = distance;
               if constexpr( WithPredecessors )
                  predecessors[ neighbor ] = current;
               visitor( neighbor, distance );
               q.push( neighbor );
            }
         }
      }
   }
   else {
      breadthFirstSearchParallel< WithPredecessors >(
         graph,
         start,
         std::forward< Visitor >( visitor ),
         distances,
         predecessors,
         deterministic,
         launchConfig,
         bypassThreshold );
   }
}

template< typename Graph, typename Vector >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig,
   double bypassThreshold )
{
   Vector dummy;
   breadthFirstSearch_impl< false >(
      graph,
      start,
      [] __cuda_callable__( typename Graph::IndexType, typename Graph::IndexType ) {},
      distances,
      dummy,
      false,
      launchConfig,
      bypassThreshold );
}

template< typename Graph, typename Vector, typename Visitor, typename Enable >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig,
   double bypassThreshold )
{
   static_assert( detail::isBfsVisitor_v< Visitor, Graph >, "BFS visitor must accept (node, distance)." );
   Vector dummy;
   breadthFirstSearch_impl< false >(
      graph, start, std::forward< Visitor >( visitor ), distances, dummy, false, launchConfig, bypassThreshold );
}

template< typename Graph, typename Vector, typename PredecessorVector >
void
breadthFirstSearchWithPredecessors(
   const Graph& graph,
   typename Graph::IndexType start,
   Vector& distances,
   PredecessorVector& predecessors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig,
   bool deterministic,
   double bypassThreshold )
{
   breadthFirstSearch_impl< true >(
      graph,
      start,
      [] __cuda_callable__( typename Graph::IndexType, typename Graph::IndexType ) {},
      distances,
      predecessors,
      deterministic,
      launchConfig,
      bypassThreshold );
}

template< typename Graph, typename Vector, typename PredecessorVector, typename Visitor, typename Enable >
void
breadthFirstSearchWithVisitorAndPredecessors(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   PredecessorVector& predecessors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig,
   bool deterministic,
   double bypassThreshold )
{
   static_assert( detail::isBfsVisitor_v< Visitor, Graph >, "BFS visitor must accept (node, distance)." );
   breadthFirstSearch_impl< true >(
      graph, start, std::forward< Visitor >( visitor ), distances, predecessors, deterministic, launchConfig, bypassThreshold );
}

}  // namespace TNL::Graphs::Algorithms
