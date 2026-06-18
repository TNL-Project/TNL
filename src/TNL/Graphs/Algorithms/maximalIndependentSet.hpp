// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <utility>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Matrices/MatrixBase.h>

#include "details/activeVertices.hpp"
#include "maximalIndependentSet.h"

namespace TNL::Graphs::Algorithms {

namespace detail {

// Deterministic pseudo-random priority for Luby's MIS algorithm.
// Uses a splitmix64-style hash to map (vertex, roundSeed, iteration) to a
// uniformly distributed 64-bit value.  The hash is deterministic so that
// every vertex computes the same priority for a given neighbor in each
// round, enabling lock-free local winner selection.
template< typename Index >
unsigned long long __cuda_callable__
maximalIndependentSetPriority( Index vertex, Index roundSeed, Index iteration )
{
   using UInt = unsigned long long;

   UInt x = static_cast< UInt >( vertex ) + 0x9e3779b97f4a7c15ULL * ( static_cast< UInt >( roundSeed ) + 1ULL );
   x ^= 0xbf58476d1ce4e5b9ULL * ( static_cast< UInt >( iteration ) + 1ULL );
   x ^= x >> 30;
   x *= 0xbf58476d1ce4e5b9ULL;
   x ^= x >> 27;
   x *= 0x94d049bb133111ebULL;
   x ^= x >> 31;
   return x;
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
maximalIndependentSetOnActiveVertices(
   const Graph& graph,
   VertexPredicate&& isActive,
   Vector& independentSet,
   typename Graph::IndexType roundSeed,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( ! Graph::isDirected(), "Maximal independent set requires an undirected graph." );

   // Deterministic Luby-style MIS on the induced subgraph given by isActive predicate and edgePredicate:
   // each round keeps local priority winners, adds them to the MIS, and removes
   // both the winners and their active neighbors from further competition.

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using SetValueType = typename Vector::ValueType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   const IndexType verticesCount = graph.getVertexCount();
   if( verticesCount == 0 ) {
      independentSet.setSize( 0 );
      return;
   }

   independentSet.setSize( verticesCount );
   independentSet = static_cast< SetValueType >( 0 );

   const auto graphView = graph.getConstView();

   IndexVector available( verticesCount, 0 );
   IndexVector candidates( verticesCount, 0 );
   IndexVector blocked( verticesCount, 0 );

   {
      auto availableView = available.getView();
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            availableView[ vertex ] = isActive( vertex ) ? 1 : 0;
         } );
   }

   IndexType iteration = 0;

   while( true ) {
      const auto availableView = available.getConstView();
      const IndexType activeCount = sum( available );

      if( activeCount == 0 )
         return;

      auto candidatesView = candidates.getView();
      const IndexType iterationRound = iteration;

      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( ! availableView[ vertex ] ) {
               candidatesView[ vertex ] = 0;
               return;
            }

            const auto priority = maximalIndependentSetPriority( vertex, roundSeed, iterationRound );
            bool wins = true;
            const auto vertexView = graphView.getVertex( vertex );

            for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
               const IndexType neighbor = vertexView.getTargetIndex( localIdx );
               if( ! availableView[ neighbor ] )
                  continue;

               const auto weight = vertexView.getEdgeWeight( localIdx );
               if( ! edgePredicate( vertex, neighbor, weight ) )
                  continue;

               const auto neighborPriority = maximalIndependentSetPriority( neighbor, roundSeed, iterationRound );
               if( neighborPriority > priority || ( neighborPriority == priority && neighbor < vertex ) ) {
                  wins = false;
                  break;
               }
            }

            candidatesView[ vertex ] = wins ? 1 : 0;
         } );

      const auto candidatesConstView = candidates.getConstView();
      const IndexType selectedThisIteration = sum( candidates );

      if( selectedThisIteration == 0 )
         throw std::logic_error( "Maximal independent set made no progress in a Luby round." );

      auto blockedView = blocked.getView();
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( ! availableView[ vertex ] ) {
               blockedView[ vertex ] = 1;
               return;
            }

            if( candidatesConstView[ vertex ] ) {
               blockedView[ vertex ] = 1;
               return;
            }

            const auto vertexView = graphView.getVertex( vertex );
            for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
               const IndexType neighbor = vertexView.getTargetIndex( localIdx );
               if( ! availableView[ neighbor ] )
                  continue;

               const auto weight = vertexView.getEdgeWeight( localIdx );
               if( ! edgePredicate( vertex, neighbor, weight ) )
                  continue;

               if( candidatesConstView[ neighbor ] ) {
                  blockedView[ vertex ] = 1;
                  return;
               }
            }

            blockedView[ vertex ] = 0;
         } );

      const auto blockedConstView = blocked.getConstView();
      auto independentSetView = independentSet.getView();
      auto availableMutableView = available.getView();
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( candidatesConstView[ vertex ] )
               independentSetView[ vertex ] = static_cast< SetValueType >( 1 );

            availableMutableView[ vertex ] = ( availableMutableView[ vertex ] && ! blockedConstView[ vertex ] ) ? 1 : 0;
         } );

      iteration++;
   }
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
bool
isMaximalIndependentSetOnActiveVertices(
   const Graph& graph,
   VertexPredicate&& isActive,
   const Vector& independentSet,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( ! Graph::isDirected(), "Maximal independent set requires an undirected graph." );

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   const IndexType verticesCount = graph.getVertexCount();
   if( independentSet.getSize() != verticesCount )
      return false;
   if( verticesCount == 0 )
      return true;

   const auto graphView = graph.getConstView();
   const auto independentSetView = independentSet.getConstView();

   return TNL::Algorithms::reduce< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) -> bool
      {
         const bool active = isActive( vertex );
         const bool isSelected = static_cast< bool >( independentSetView[ vertex ] );
         if( ! active )
            return ! isSelected;

         bool hasSelectedNeighbor = false;
         const auto vertexView = graphView.getVertex( vertex );
         for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
            const IndexType neighbor = vertexView.getTargetIndex( localIdx );
            if( ! isActive( neighbor ) )
               continue;

            const auto weight = vertexView.getEdgeWeight( localIdx );
            if( ! edgePredicate( vertex, neighbor, weight ) )
               continue;

            if( static_cast< bool >( independentSetView[ neighbor ] ) ) {
               if( isSelected )
                  return false;
               hasSelectedNeighbor = true;
            }
         }

         return isSelected || hasSelectedNeighbor;
      },
      LogicalAnd{} );
}

}  // namespace detail

template< typename Graph, typename Vector >
void
maximalIndependentSet( const Graph& graph, Vector& independentSet, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   detail::maximalIndependentSetOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      independentSet,
      0,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgePredicate, typename >
void
maximalIndependentSet(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   detail::maximalIndependentSetOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      independentSet,
      0,
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
void
maximalIndependentSet(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   detail::maximalIndependentSetOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      independentSet,
      0,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename EdgePredicate, typename >
void
maximalIndependentSet(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   detail::maximalIndependentSetOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      independentSet,
      0,
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector >
void
maximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   detail::maximalIndependentSetOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      independentSet,
      0,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
maximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::maximalIndependentSetOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      independentSet,
      0,
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename Vector >
bool
isMaximalIndependentSet(
   const Graph& graph,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   return detail::isMaximalIndependentSetOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      independentSet,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgePredicate, typename >
bool
isMaximalIndependentSet(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   return detail::isMaximalIndependentSetOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      independentSet,
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
bool
isMaximalIndependentSet(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   return detail::isMaximalIndependentSetOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      independentSet,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename EdgePredicate, typename >
bool
isMaximalIndependentSet(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   return detail::isMaximalIndependentSetOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      independentSet,
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector >
bool
isMaximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   return detail::isMaximalIndependentSetOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      independentSet,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
bool
isMaximalIndependentSetIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return detail::isMaximalIndependentSetOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      independentSet,
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
