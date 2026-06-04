// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Assert.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>
#include <TNL/Graphs/Algorithms/maximalIndependentSet.h>
#include <TNL/Matrices/MatrixBase.h>

#include "graphColoring.h"

namespace TNL::Graphs::Algorithms {

namespace detail {

template< typename ColorType >
constexpr ColorType
maskedInactiveColor()
{
   static_assert(
      std::is_signed_v< ColorType > || std::is_floating_point_v< ColorType >,
      "Masked graph coloring requires a signed or floating-point color type because inactive vertices are marked by -1." );

   return static_cast< ColorType >( -1 );
}

template< typename Graph, typename ActiveVector, typename Vector >
bool
isProperlyColoredOnActiveVertices(
   const Graph& graph,
   const ActiveVector& activeVertices,
   const Vector& colors,
   const typename Vector::ValueType minimumColor,
   const typename Vector::ValueType inactiveColor )
{
   static_assert( ! Graph::isDirected(), "Graph coloring requires an undirected graph." );

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ColorType = typename Vector::ValueType;

   const IndexType verticesCount = graph.getVertexCount();
   if( activeVertices.getSize() != verticesCount || colors.getSize() != verticesCount )
      return false;
   if( verticesCount == 0 )
      return true;

   const auto graphView = graph.getConstView();
   const auto activeVerticesView = activeVertices.getConstView();
   const auto colorsView = colors.getConstView();

   return TNL::Algorithms::reduce< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) -> bool
      {
         const bool isActive = static_cast< bool >( activeVerticesView[ vertex ] );
         const ColorType color = colorsView[ vertex ];
         if( ! isActive )
            return color == inactiveColor;

         if( color < minimumColor )
            return false;

         const auto vertexView = graphView.getVertex( vertex );
         for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
            const IndexType neighbor = vertexView.getTargetIndex( localIdx );
            if( ! activeVerticesView[ neighbor ] )
               continue;

            if( colorsView[ neighbor ] == color )
               return false;
         }

         return true;
      },
      TNL::LogicalAnd{} );
}

template< typename Graph, typename ActiveVector, typename Vector >
void
finalizeZeroBasedColoringOnActiveVertices(
   const Graph& graph,
   const ActiveVector& activeVertices,
   Vector& colors,
   const typename Vector::ValueType inactiveColor )
{
   TNL_ASSERT_TRUE(
      isProperlyColoredOnActiveVertices(
         graph, activeVertices, colors, static_cast< typename Vector::ValueType >( 1 ), inactiveColor ),
      "Internal graph coloring must be proper before conversion to zero-based labels." );

   using IndexType = typename Graph::IndexType;

   const IndexType verticesCount = graph.getVertexCount();
   if( verticesCount == 0 )
      return;

   colors -= activeVertices;
}

template< typename Graph, typename ActiveVector, typename Vector >
void
graphColoringOnActiveVerticesSequential(
   const Graph& graph,
   const ActiveVector& activeVertices,
   Vector& colors,
   const typename Vector::ValueType inactiveColor )
{
   static_assert( ! Graph::isDirected(), "Graph coloring requires an undirected graph." );

   using IndexType = typename Graph::IndexType;
   using ColorType = typename Vector::ValueType;

   const IndexType verticesCount = graph.getVertexCount();

   const auto graphView = graph.getConstView();
   const auto activeVerticesView = activeVertices.getConstView();
   auto colorsView = colors.getView();

   // On the sequential device, a plain first-fit pass avoids speculative rounds
   // and colors each active vertex immediately in the chosen traversal order.
   for( IndexType vertex = 0; vertex < verticesCount; vertex++ ) {
      if( ! activeVerticesView[ vertex ] )
         continue;

      ColorType candidate = static_cast< ColorType >( 1 );
      const auto vertexView = graphView.getVertex( vertex );

      while( true ) {
         bool blocked = false;

         for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
            const IndexType neighbor = vertexView.getTargetIndex( localIdx );
            if( ! activeVerticesView[ neighbor ] )
               continue;

            if( colorsView[ neighbor ] == candidate ) {
               blocked = true;
               candidate++;
               break;
            }
         }

         if( ! blocked ) {
            colorsView[ vertex ] = candidate;
            break;
         }
      }
   }

   finalizeZeroBasedColoringOnActiveVertices( graph, activeVertices, colors, inactiveColor );
}

template< typename Graph, typename ActiveVector, typename Vector >
void
graphColoringOnActiveVertices(
   const Graph& graph,
   const ActiveVector& activeVertices,
   Vector& colors,
   const typename Vector::ValueType inactiveColor )
{
   static_assert( ! Graph::isDirected(), "Graph coloring requires an undirected graph." );

   // Parallel speculative greedy coloring on the induced subgraph given by
   // activeVertices mask: each active uncolored vertex proposes the smallest
   // currently safe color, conflicting proposals are filtered by priority, and
   // only the winners commit their color in the current round.

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ColorType = typename Vector::ValueType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;
   using ColorVector = Containers::Vector< ColorType, DeviceType, IndexType >;

   const IndexType verticesCount = graph.getVertexCount();
   if( activeVertices.getSize() != verticesCount )
      throw std::invalid_argument( "Active-vertex mask size must match the graph size." );

   colors.setSize( verticesCount );
   if( verticesCount == 0 )
      return;

   // Start with all active vertices uncolored and keep inactive ones marked out.
   colors = inactiveColor + activeVertices * ( static_cast< ColorType >( 0 ) - inactiveColor );

   if constexpr( std::is_same_v< DeviceType, Devices::Sequential > ) {
      graphColoringOnActiveVerticesSequential( graph, activeVertices, colors, inactiveColor );
      return;
   }

   const auto activeVerticesView = activeVertices.getConstView();
   const auto graphView = graph.getConstView();

   IndexVector degrees( verticesCount, 0 );
   auto degreesView = degrees.getView();

   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) mutable
      {
         degreesView[ vertex ] = graphView.getVertex( vertex ).getDegree();
      } );

   // Stores the tentative color chosen for each vertex in the current round.
   ColorVector proposedColors( verticesCount, 0 );
   // Marks vertices whose proposal wins the current conflict-resolution round.
   IndexVector keepColor( verticesCount, 0 );

   while( true ) {
      // Stop once every active vertex has received a color.
      const IndexType uncoloredVertices = TNL::sum( activeVertices * TNL::equalTo( colors, static_cast< ColorType >( 0 ) ) );

      if( uncoloredVertices == 0 )
         break;

      const auto colorsView = colors.getConstView();
      auto proposedColorsView = proposedColors.getView();
      // Every active uncolored vertex proposes its smallest currently admissible color.
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( ! activeVerticesView[ vertex ] )
               return;

            if( colorsView[ vertex ] != static_cast< ColorType >( 0 ) )
               return;

            // Greedily propose the smallest color that is already safe with respect to colored neighbors.
            ColorType candidate = static_cast< ColorType >( 1 );
            while( true ) {
               bool blocked = false;
               const auto vertexView = graphView.getVertex( vertex );

               for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
                  const IndexType neighbor = vertexView.getTargetIndex( localIdx );
                  if( ! activeVerticesView[ neighbor ] )
                     continue;

                  if( colorsView[ neighbor ] == candidate ) {
                     blocked = true;
                     candidate++;
                     break;
                  }
               }

               if( ! blocked ) {
                  proposedColorsView[ vertex ] = candidate;
                  return;
               }
            }
         } );

      const auto proposedColorsConstView = proposedColors.getConstView();
      const auto degreesConstView = degrees.getConstView();
      auto keepColorView = keepColor.getView();

      // Conflicting neighbors may propose the same color; keep only the local winner and retry the losers later.
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( ! activeVerticesView[ vertex ] ) {
               keepColorView[ vertex ] = 0;
               return;
            }

            if( colorsView[ vertex ] != static_cast< ColorType >( 0 ) ) {
               keepColorView[ vertex ] = 0;
               return;
            }

            const ColorType candidate = proposedColorsConstView[ vertex ];
            bool keep = true;
            const auto vertexView = graphView.getVertex( vertex );

            for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
               const IndexType neighbor = vertexView.getTargetIndex( localIdx );
               if( ! activeVerticesView[ neighbor ] )
                  continue;

               // Already colored neighbors were handled during proposal generation,
               // so only same-round competitors matter here.
               if( colorsView[ neighbor ] != static_cast< ColorType >( 0 ) )
                  continue;

               if( proposedColorsConstView[ neighbor ] != candidate )
                  continue;

               const bool neighborHasHigherPriority =
                  degreesConstView[ neighbor ] > degreesConstView[ vertex ]
                  || ( degreesConstView[ neighbor ] == degreesConstView[ vertex ] && neighbor < vertex );
               if( neighborHasHigherPriority ) {
                  keep = false;
                  break;
               }
            }

            keepColorView[ vertex ] = keep;
         } );

      const IndexType coloredThisRound = TNL::sum( keepColor );

      if( coloredThisRound == 0 )
         throw std::logic_error( "Graph coloring made no progress in a round." );

      // Commit only the winning proposals and leave the remaining active vertices for the next round.
      colors += keepColor * proposedColors;
   }

   finalizeZeroBasedColoringOnActiveVertices( graph, activeVertices, colors, inactiveColor );
}

template< typename Graph, typename ActiveVector, typename Vector >
void
graphColoringLubiOnActiveVertices(
   const Graph& graph,
   const ActiveVector& activeVertices,
   Vector& colors,
   const typename Vector::ValueType inactiveColor )
{
   static_assert( ! Graph::isDirected(), "Graph coloring requires an undirected graph." );

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ColorType = typename Vector::ValueType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   const IndexType verticesCount = graph.getVertexCount();
   if( activeVertices.getSize() != verticesCount )
      throw std::invalid_argument( "Active-vertex mask size must match the graph size." );

   if( verticesCount == 0 )
      return;
   colors.setSize( verticesCount );

   colors = inactiveColor + activeVertices * ( static_cast< ColorType >( 0 ) - inactiveColor );

   IndexVector remaining( verticesCount, 0 );
   IndexVector selected( verticesCount, 0 );
   remaining = activeVertices;

   ColorType currentColor = static_cast< ColorType >( 1 );

   while( true ) {
      const IndexType uncoloredVertices = TNL::sum( remaining );

      if( uncoloredVertices == 0 )
         break;

      detail::maximalIndependentSetOnActiveVertices( graph, remaining, selected, static_cast< IndexType >( currentColor ) );

      // Each color class is one maximal independent set of the still-uncolored subgraph.
      colors += currentColor * selected;
      remaining -= selected;

      currentColor++;
   }

   finalizeZeroBasedColoringOnActiveVertices( graph, activeVertices, colors, inactiveColor );
}

}  // namespace detail

template< typename Graph, typename Vector >
void
graphColoring( const Graph& graph, Vector& colors )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateAllVertices( graph, activeVertices );
   detail::graphColoringOnActiveVertices( graph, activeVertices, colors, static_cast< typename Vector::ValueType >( 0 ) );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
void
graphColoring( const Graph& graph, const VertexIndexes& vertexIndexes, Vector& colors )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   detail::graphColoringOnActiveVertices(
      graph, activeVertices, colors, detail::maskedInactiveColor< typename Vector::ValueType >() );
}

template< typename Graph, typename VertexPredicate, typename Vector >
void
graphColoringIf( const Graph& graph, VertexPredicate&& vertexPredicate, Vector& colors )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateVerticesIf( graph, std::forward< VertexPredicate >( vertexPredicate ), activeVertices );
   detail::graphColoringOnActiveVertices(
      graph, activeVertices, colors, detail::maskedInactiveColor< typename Vector::ValueType >() );
}

template< typename Graph, typename Vector >
void
graphColoringLubi( const Graph& graph, Vector& colors )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateAllVertices( graph, activeVertices );
   detail::graphColoringLubiOnActiveVertices( graph, activeVertices, colors, static_cast< typename Vector::ValueType >( 0 ) );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
void
graphColoringLubi( const Graph& graph, const VertexIndexes& vertexIndexes, Vector& colors )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   detail::graphColoringLubiOnActiveVertices(
      graph, activeVertices, colors, detail::maskedInactiveColor< typename Vector::ValueType >() );
}

template< typename Graph, typename VertexPredicate, typename Vector >
void
graphColoringLubiIf( const Graph& graph, VertexPredicate&& vertexPredicate, Vector& colors )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateVerticesIf( graph, std::forward< VertexPredicate >( vertexPredicate ), activeVertices );
   detail::graphColoringLubiOnActiveVertices(
      graph, activeVertices, colors, detail::maskedInactiveColor< typename Vector::ValueType >() );
}

template< typename Graph, typename Vector >
bool
isProperlyColored( const Graph& graph, const Vector& colors )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateAllVertices( graph, activeVertices );
   return detail::isProperlyColoredOnActiveVertices(
      graph,
      activeVertices,
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      static_cast< typename Vector::ValueType >( 0 ) );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
bool
isProperlyColored( const Graph& graph, const VertexIndexes& vertexIndexes, const Vector& colors )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   return detail::isProperlyColoredOnActiveVertices(
      graph,
      activeVertices,
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      detail::maskedInactiveColor< typename Vector::ValueType >() );
}

template< typename Graph, typename VertexPredicate, typename Vector >
bool
isProperlyColoredIf( const Graph& graph, VertexPredicate&& vertexPredicate, const Vector& colors )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateVerticesIf( graph, std::forward< VertexPredicate >( vertexPredicate ), activeVertices );
   return detail::isProperlyColoredOnActiveVertices(
      graph,
      activeVertices,
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      detail::maskedInactiveColor< typename Vector::ValueType >() );
}

}  // namespace TNL::Graphs::Algorithms
