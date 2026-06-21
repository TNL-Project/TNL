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
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
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

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
bool
isProperlyColoredOnActiveVertices(
   const Graph& graph,
   VertexPredicate&& isActive,
   const Vector& colors,
   const typename Vector::ValueType minimumColor,
   const typename Vector::ValueType inactiveColor,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( ! Graph::isDirected(), "Graph coloring requires an undirected graph." );

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ColorType = typename Vector::ValueType;

   const IndexType verticesCount = graph.getVertexCount();
   if( colors.getSize() != verticesCount )
      return false;
   if( verticesCount == 0 )
      return true;

   const auto graphView = graph.getConstView();
   const auto colorsView = colors.getConstView();

   return TNL::Algorithms::reduce< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) -> bool
      {
         const bool active = isActive( vertex );
         const ColorType color = colorsView[ vertex ];
         if( ! active )
            return color == inactiveColor;

         if( color < minimumColor )
            return false;

         const auto vertexView = graphView.getVertex( vertex );
         for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
            const IndexType neighbor = vertexView.getTargetIndex( localIdx );
            if( ! isActive( neighbor ) )
               continue;

            const auto weight = vertexView.getEdgeWeight( localIdx );
            if( ! edgePredicate( vertex, neighbor, weight ) )
               continue;

            if( colorsView[ neighbor ] == color )
               return false;
         }

         return true;
      },
      LogicalAnd{} );
}

// Convert 1-based colors to 0-based by subtracting 1 from each active vertex.
// Internally, colors start at 1 (0 means "uncolored"); this function shifts
// them down so the final result uses 0, 1, 2, ...
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
finalizeZeroBasedColoringOnActiveVertices(
   const Graph& graph,
   VertexPredicate&& isActive,
   Vector& colors,
   const typename Vector::ValueType inactiveColor,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   TNL_ASSERT_TRUE(
      isProperlyColoredOnActiveVertices(
         graph, isActive, colors, static_cast< typename Vector::ValueType >( 1 ), inactiveColor, edgePredicate, launchConfig ),
      "Internal graph coloring must be proper before conversion to zero-based labels." );

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   const IndexType verticesCount = graph.getVertexCount();
   if( verticesCount == 0 )
      return;

   auto colorsView = colors.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      verticesCount,
      [ = ] __cuda_callable__( IndexType vertex ) mutable
      {
         if( isActive( vertex ) )
            colorsView[ vertex ] -= static_cast< typename Vector::ValueType >( 1 );
      } );
}

// Sequential greedy coloring: process vertices in index order, assign each
// vertex the smallest color not used by any already-colored neighbor.
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
graphColoringOnActiveVerticesSequential(
   const Graph& graph,
   VertexPredicate&& isActive,
   Vector& colors,
   const typename Vector::ValueType inactiveColor,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( ! Graph::isDirected(), "Graph coloring requires an undirected graph." );

   using IndexType = typename Graph::IndexType;
   using ColorType = typename Vector::ValueType;

   const IndexType verticesCount = graph.getVertexCount();

   const auto graphView = graph.getConstView();
   auto colorsView = colors.getView();

   for( IndexType vertex = 0; vertex < verticesCount; vertex++ ) {
      if( ! isActive( vertex ) )
         continue;

      ColorType candidate = static_cast< ColorType >( 1 );
      const auto vertexView = graphView.getVertex( vertex );

      while( true ) {
         bool blocked = false;

         for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
            const IndexType neighbor = vertexView.getTargetIndex( localIdx );
            if( ! isActive( neighbor ) )
               continue;

            const auto weight = vertexView.getEdgeWeight( localIdx );
            if( ! edgePredicate( vertex, neighbor, weight ) )
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

   finalizeZeroBasedColoringOnActiveVertices( graph, isActive, colors, inactiveColor, edgePredicate, launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
graphColoringOnActiveVertices(
   const Graph& graph,
   VertexPredicate&& isActive,
   Vector& colors,
   const typename Vector::ValueType inactiveColor,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( ! Graph::isDirected(), "Graph coloring requires an undirected graph." );

   // Parallel speculative greedy coloring on the induced subgraph given by
   // isActive predicate: each active uncolored vertex proposes the smallest
   // currently safe color, conflicting proposals are filtered by priority, and
   // only the winners commit their color in the current round.

   using DeviceType = typename Graph::DeviceType;
   static_assert(
      std::is_same_v< DeviceType, Devices::Sequential > || ! Graph::AdjacencyMatrixType::MatrixType::isSymmetric(),
      "Parallel graph coloring requires a general (non-symmetric) adjacency matrix. "
      "Use a Sequential device for symmetric matrices." );
   using IndexType = typename Graph::IndexType;
   using ColorType = typename Vector::ValueType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;
   using ColorVector = Containers::Vector< ColorType, DeviceType, IndexType >;

   const IndexType verticesCount = graph.getVertexCount();
   colors.setSize( verticesCount );
   if( verticesCount == 0 )
      return;

   {
      auto colorsInitView = colors.getView();
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            colorsInitView[ vertex ] = isActive( vertex ) ? static_cast< ColorType >( 0 ) : inactiveColor;
         } );
   }

   if constexpr( std::is_same_v< DeviceType, Devices::Sequential > ) {
      graphColoringOnActiveVerticesSequential( graph, isActive, colors, inactiveColor, edgePredicate, launchConfig );
      return;
   }

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

   ColorVector proposedColors( verticesCount, 0 );
   IndexVector keepColor( verticesCount, 0 );

   while( true ) {
      const auto colorsView = colors.getConstView();
      const IndexType uncoloredVertices = TNL::Algorithms::reduce< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) -> IndexType
         {
            return ( isActive( vertex ) && colorsView[ vertex ] == static_cast< ColorType >( 0 ) ) ? 1 : 0;
         },
         Plus{} );

      if( uncoloredVertices == 0 )
         break;

      auto proposedColorsView = proposedColors.getView();
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( ! isActive( vertex ) )
               return;

            if( colorsView[ vertex ] != static_cast< ColorType >( 0 ) )
               return;

            ColorType candidate = static_cast< ColorType >( 1 );
            while( true ) {
               bool blocked = false;
               const auto vertexView = graphView.getVertex( vertex );

               for( IndexType localIdx = 0; localIdx < vertexView.getDegree(); localIdx++ ) {
                  const IndexType neighbor = vertexView.getTargetIndex( localIdx );
                  if( ! isActive( neighbor ) )
                     continue;

                  const auto weight = vertexView.getEdgeWeight( localIdx );
                  if( ! edgePredicate( vertex, neighbor, weight ) )
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

      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            if( ! isActive( vertex ) ) {
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
               if( ! isActive( neighbor ) )
                  continue;

               const auto weight = vertexView.getEdgeWeight( localIdx );
               if( ! edgePredicate( vertex, neighbor, weight ) )
                  continue;

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

      const IndexType coloredThisRound = sum( keepColor );

      if( coloredThisRound == 0 )
         throw std::logic_error( "Graph coloring made no progress in a round." );

      colors += keepColor * proposedColors;
   }

   finalizeZeroBasedColoringOnActiveVertices( graph, isActive, colors, inactiveColor, edgePredicate, launchConfig );
}

// Luby-based coloring: repeatedly find a maximal independent set among
// uncolored vertices and assign all MIS members the same color.  Each
// iteration produces one color class.
template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
graphColoringLubyOnActiveVertices(
   const Graph& graph,
   VertexPredicate&& isActive,
   Vector& colors,
   const typename Vector::ValueType inactiveColor,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert( ! Graph::isDirected(), "Graph coloring requires an undirected graph." );

   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ColorType = typename Vector::ValueType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   const IndexType verticesCount = graph.getVertexCount();
   if( verticesCount == 0 )
      return;
   colors.setSize( verticesCount );

   {
      auto colorsInitView = colors.getView();
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            colorsInitView[ vertex ] = isActive( vertex ) ? static_cast< ColorType >( 0 ) : inactiveColor;
         } );
   }

   IndexVector remaining( verticesCount, 0 );
   IndexVector selected( verticesCount, 0 );
   {
      auto remainingView = remaining.getView();
      TNL::Algorithms::parallelFor< DeviceType >(
         0,
         verticesCount,
         [ = ] __cuda_callable__( IndexType vertex ) mutable
         {
            remainingView[ vertex ] = isActive( vertex ) ? 1 : 0;
         } );
   }

   ColorType currentColor = static_cast< ColorType >( 1 );

   while( true ) {
      const IndexType uncoloredVertices = sum( remaining );

      if( uncoloredVertices == 0 )
         break;

      const auto remainingConstView = remaining.getConstView();
      detail::maximalIndependentSetOnActiveVertices(
         graph,
         [ = ] __cuda_callable__( IndexType vertex )
         {
            return static_cast< bool >( remainingConstView[ vertex ] );
         },
         selected,
         static_cast< IndexType >( currentColor ),
         edgePredicate,
         launchConfig );

      colors += currentColor * selected;
      remaining -= selected;

      currentColor++;
   }

   finalizeZeroBasedColoringOnActiveVertices( graph, isActive, colors, inactiveColor, edgePredicate, launchConfig );
}

}  // namespace detail

template< typename Graph, typename Vector >
void
graphColoring( const Graph& graph, Vector& colors, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   detail::graphColoringOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgePredicate, typename Enable >
void
graphColoring(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   detail::graphColoringOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename Enable >
void
graphColoring(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   detail::graphColoringOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      colors,
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename EdgePredicate, typename Enable >
void
graphColoring(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   detail::graphColoringOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      colors,
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector >
void
graphColoringIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   detail::graphColoringOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      colors,
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
graphColoringIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::graphColoringOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      colors,
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename Vector >
void
graphColoringLuby( const Graph& graph, Vector& colors, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   detail::graphColoringLubyOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgePredicate, typename Enable >
void
graphColoringLuby(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   detail::graphColoringLubyOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename Enable >
void
graphColoringLuby(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   detail::graphColoringLubyOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      colors,
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename EdgePredicate, typename Enable >
void
graphColoringLuby(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   detail::graphColoringLubyOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      colors,
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector >
void
graphColoringLubyIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   detail::graphColoringLubyOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      colors,
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
void
graphColoringLubyIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::graphColoringLubyOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      colors,
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename Vector >
bool
isProperlyColored( const Graph& graph, const Vector& colors, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   return detail::isProperlyColoredOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      static_cast< typename Vector::ValueType >( 0 ),
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename Vector, typename EdgePredicate, typename Enable >
bool
isProperlyColored(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   return detail::isProperlyColoredOnActiveVertices(
      graph,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      static_cast< typename Vector::ValueType >( 0 ),
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename Enable >
bool
isProperlyColored(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   return detail::isProperlyColoredOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename EdgePredicate, typename Enable >
bool
isProperlyColored(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   return detail::isProperlyColoredOnActiveVertices(
      graph,
      [ = ] __cuda_callable__( IndexType vertex )
      {
         return static_cast< bool >( activeVerticesView[ vertex ] );
      },
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector >
bool
isProperlyColoredIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   return detail::isProperlyColoredOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      [] __cuda_callable__( IndexType, IndexType, typename Graph::ValueType )
      {
         return true;
      },
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector, typename EdgePredicate >
bool
isProperlyColoredIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   const Vector& colors,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return detail::isProperlyColoredOnActiveVertices(
      graph,
      std::forward< VertexPredicate >( vertexPredicate ),
      colors,
      static_cast< typename Vector::ValueType >( 0 ),
      detail::maskedInactiveColor< typename Vector::ValueType >(),
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
