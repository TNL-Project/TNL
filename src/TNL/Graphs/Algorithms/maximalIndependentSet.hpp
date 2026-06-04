// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <utility>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>
#include <TNL/Matrices/MatrixBase.h>

#include "details/activeVertices.hpp"
#include "maximalIndependentSet.h"

namespace TNL::Graphs::Algorithms {

namespace detail {

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

template< typename Graph, typename ActiveVector, typename Vector >
void
maximalIndependentSetOnActiveVertices(
   const Graph& graph,
   const ActiveVector& activeVertices,
   Vector& independentSet,
   typename Graph::IndexType roundSeed = 0 )
{
   static_assert( ! Graph::isDirected(), "Maximal independent set requires an undirected graph." );

   // Deterministic Luby-style MIS on the induced subgraph given by activeVertices:
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

   if( activeVertices.getSize() != verticesCount )
      throw std::invalid_argument( "Active-vertex mask size must match the graph size." );

   independentSet.setSize( verticesCount );
   independentSet = static_cast< SetValueType >( 0 );

   const auto graphView = graph.getConstView();

   // Vertices still allowed to compete for the current MIS.
   IndexVector available( verticesCount, 0 );
   // Local winners of the current priority round.
   IndexVector candidates( verticesCount, 0 );
   // Vertices removed from further competition in this MIS build.
   IndexVector blocked( verticesCount, 0 );

   available = activeVertices;

   IndexType iteration = 0;

   while( true ) {
      const auto available_view = available.getConstView();
      const IndexType activeCount = TNL::sum( available );

      if( activeCount == 0 )
         return;

      auto candidatesView = candidates.getView();
      const IndexType iterationRound = iteration;

      // Each round picks one priority-maximal independent set from the still-active vertices.
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( ! available_view[ vertex ] ) {
               candidatesView[ vertex ] = 0;
               return;
            }

            const auto priority = maximalIndependentSetPriority( vertex, roundSeed, iterationRound );
            bool wins = true;
            const auto vertexView = graphView.getVertex( vertex );

            for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
               const IndexType neighbor = vertexView.getTargetIndex( localIdx );
               if( ! available_view[ neighbor ] )
                  continue;

               const auto neighborPriority = maximalIndependentSetPriority( neighbor, roundSeed, iterationRound );
               if( neighborPriority > priority || ( neighborPriority == priority && neighbor < vertex ) ) {
                  wins = false;
                  break;
               }
            }

            candidatesView[ vertex ] = wins ? 1 : 0;
         } );

      const auto candidates_view = candidates.getConstView();
      const IndexType selectedThisIteration = TNL::sum( candidates );

      if( selectedThisIteration == 0 )
         throw std::logic_error( "Maximal independent set made no progress in a Luby round." );

      auto blocked_view = blocked.getView();
      // Winners and their neighbors leave the active set, so the accumulated result stays independent.
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( ! available_view[ vertex ] ) {
               blocked_view[ vertex ] = 1;
               return;
            }

            if( candidates_view[ vertex ] ) {
               blocked_view[ vertex ] = 1;
               return;
            }

            const auto vertexView = graphView.getVertex( vertex );
            for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
               const IndexType neighbor = vertexView.getTargetIndex( localIdx );
               if( ! available_view[ neighbor ] )
                  continue;

               if( candidates_view[ neighbor ] ) {
                  blocked_view[ vertex ] = 1;
                  return;
               }
            }

            blocked_view[ vertex ] = 0;
         } );

      const auto blockedConstView = blocked.getConstView();
      auto independentSet_view = independentSet.getView();
      auto availableMutable_view = available.getView();
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( candidates_view[ vertex ] )
               independentSet_view[ vertex ] = static_cast< SetValueType >( 1 );

            availableMutable_view[ vertex ] = ( availableMutable_view[ vertex ] && ! blockedConstView[ vertex ] ) ? 1 : 0;
         } );

      iteration++;
   }
}

template< typename Graph, typename ActiveVector, typename Vector >
bool
isMaximalIndependentSetOnActiveVertices( const Graph& graph, const ActiveVector& activeVertices, const Vector& independentSet )
{
   static_assert( ! Graph::isDirected(), "Maximal independent set requires an undirected graph." );

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   const IndexType verticesCount = graph.getVertexCount();
   if( activeVertices.getSize() != verticesCount || independentSet.getSize() != verticesCount )
      return false;
   if( verticesCount == 0 )
      return true;

   const auto graphView = graph.getConstView();
   const auto activeVerticesView = activeVertices.getConstView();
   const auto independentSetView = independentSet.getConstView();

   return TNL::Algorithms::reduce< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) -> bool
      {
         const bool isActive = static_cast< bool >( activeVerticesView[ vertex ] );
         const bool isSelected = static_cast< bool >( independentSetView[ vertex ] );
         if( ! isActive )
            return ! isSelected;

         bool hasSelectedNeighbor = false;
         const auto vertexView = graphView.getVertex( vertex );
         for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
            const IndexType neighbor = vertexView.getTargetIndex( localIdx );
            if( ! activeVerticesView[ neighbor ] )
               continue;

            if( static_cast< bool >( independentSetView[ neighbor ] ) ) {
               if( isSelected )
                  return false;
               hasSelectedNeighbor = true;
            }
         }

         return isSelected || hasSelectedNeighbor;
      },
      TNL::LogicalAnd{} );
}

}  // namespace detail

template< typename Graph, typename Vector >
void
maximalIndependentSet( const Graph& graph, Vector& independentSet )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateAllVertices( graph, activeVertices );
   detail::maximalIndependentSetOnActiveVertices( graph, activeVertices, independentSet );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
void
maximalIndependentSet( const Graph& graph, const VertexIndexes& vertexIndexes, Vector& independentSet )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   detail::maximalIndependentSetOnActiveVertices( graph, activeVertices, independentSet );
}

template< typename Graph, typename VertexPredicate, typename Vector >
void
maximalIndependentSetIf( const Graph& graph, VertexPredicate&& vertexPredicate, Vector& independentSet )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateVerticesIf( graph, std::forward< VertexPredicate >( vertexPredicate ), activeVertices );
   detail::maximalIndependentSetOnActiveVertices( graph, activeVertices, independentSet );
}

template< typename Graph, typename Vector >
bool
isMaximalIndependentSet( const Graph& graph, const Vector& independentSet )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateAllVertices( graph, activeVertices );
   return detail::isMaximalIndependentSetOnActiveVertices( graph, activeVertices, independentSet );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
bool
isMaximalIndependentSet( const Graph& graph, const VertexIndexes& vertexIndexes, const Vector& independentSet )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   return detail::isMaximalIndependentSetOnActiveVertices( graph, activeVertices, independentSet );
}

template< typename Graph, typename VertexPredicate, typename Vector >
bool
isMaximalIndependentSetIf( const Graph& graph, VertexPredicate&& vertexPredicate, const Vector& independentSet )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateVerticesIf( graph, std::forward< VertexPredicate >( vertexPredicate ), activeVertices );
   return detail::isMaximalIndependentSetOnActiveVertices( graph, activeVertices, independentSet );
}

}  // namespace TNL::Graphs::Algorithms
