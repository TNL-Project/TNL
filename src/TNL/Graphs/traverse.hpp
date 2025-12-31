// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "detail/TraversingOperations.h"

namespace TNL::Graphs {

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Function >
void
forEdges( Graph& graph,
          IndexBegin begin,
          IndexEnd end,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::TraversingOperations< typename Graph::ViewType >::forEdges(
      graph_view, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Function >
void
forEdges( const Graph& graph,
          IndexBegin begin,
          IndexEnd end,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Graph::ConstViewType >::forEdges(
      graph.getConstView(), begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Function >
void
forAllEdges( Graph& graph, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   auto graph_view = graph.getView();
   detail::TraversingOperations< typename Graph::ViewType >::forEdges(
      graph_view, (IndexType) 0, graph.getVertexCount(), std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Function >
void
forAllEdges( const Graph& graph, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   detail::TraversingOperations< typename Graph::ConstViewType >::forEdges(
      graph.getConstView(), (IndexType) 0, graph.getVertexCount(), std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forEdges( Graph& graph,
          const Array& vertexIndexes,
          IndexBegin begin,
          IndexEnd end,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::TraversingOperations< typename Graph::ViewType >::forEdges(
      graph_view, vertexIndexes, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forEdges( const Graph& graph,
          const Array& vertexIndexes,
          IndexBegin begin,
          IndexEnd end,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Graph::ConstViewType >::forEdges(
      graph.getConstView(), vertexIndexes, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Array, typename Function >
void
forEdges( Graph& graph,
          const Array& vertexIndexes,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   auto graph_view = graph.getView();
   detail::TraversingOperations< typename Graph::ViewType >::forEdges(
      graph_view, vertexIndexes, (IndexType) 0, vertexIndexes.getSize(), std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Array, typename Function >
void
forEdges( const Graph& graph,
          const Array& vertexIndexes,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   detail::TraversingOperations< typename Graph::ConstViewType >::forEdges( graph.getConstView(),
                                                                            vertexIndexes,
                                                                            (IndexType) 0,
                                                                            vertexIndexes.getSize(),
                                                                            std::forward< Function >( function ),
                                                                            launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forEdgesIf( Graph& graph,
            IndexBegin begin,
            IndexEnd end,
            Condition&& condition,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::TraversingOperations< typename Graph::ViewType >::forEdgesIf(
      graph_view, begin, end, std::forward< Condition >( condition ), std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forEdgesIf( const Graph& graph,
            IndexBegin begin,
            IndexEnd end,
            Condition&& condition,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Graph::ConstViewType >::forEdgesIf( graph.getConstView(),
                                                                              begin,
                                                                              end,
                                                                              std::forward< Condition >( condition ),
                                                                              std::forward< Function >( function ),
                                                                              launchConfig );
}

template< typename Graph, typename Condition, typename Function >
void
forAllEdgesIf( Graph& graph,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   auto graph_view = graph.getView();
   detail::TraversingOperations< typename Graph::ViewType >::forEdgesIf( graph_view,
                                                                         (IndexType) 0,
                                                                         graph.getVertexCount(),
                                                                         std::forward< Condition >( condition ),
                                                                         std::forward< Function >( function ),
                                                                         launchConfig );
}

template< typename Graph, typename Condition, typename Function >
void
forAllEdgesIf( const Graph& graph,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   detail::TraversingOperations< typename Graph::ConstViewType >::forEdgesIf( graph.getConstView(),
                                                                              (IndexType) 0,
                                                                              graph.getVertexCount(),
                                                                              std::forward< Condition >( condition ),
                                                                              std::forward< Function >( function ),
                                                                              launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forVertices( Graph& graph,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::TraversingOperations< typename Graph::ViewType >::forVertices(
      graph_view, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forVertices( const Graph& graph,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Graph::ConstViewType >::forVertices(
      graph.getConstView(), begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Function >
void
forAllVertices( Graph& graph, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   auto graph_view = graph.getView();
   forVertices( graph_view, (IndexType) 0, graph.getVertexCount(), std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Function >
void
forAllVertices( const Graph& graph, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   forVertices(
      graph.getConstView(), (IndexType) 0, graph.getVertexCount(), std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Array, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forVertices( Graph& graph,
             const Array& vertexIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::TraversingOperations< typename Graph::ViewType >::forVertices(
      graph_view, vertexIndexes, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Array, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forVertices( const Graph& graph,
             const Array& vertexIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Graph::ConstViewType >::forVertices(
      graph.getConstView(), vertexIndexes, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Array, typename Function, typename T >
void
forVertices( Graph& graph,
             const Array& vertexIndexes,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   auto graph_view = graph.getView();
   forVertices(
      graph_view, vertexIndexes, (IndexType) 0, vertexIndexes.getSize(), std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename Array, typename Function, typename T >
void
forVertices( const Graph& graph,
             const Array& vertexIndexes,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   forVertices(
      graph, vertexIndexes, (IndexType) 0, vertexIndexes.getSize(), std::forward< Function >( function ), launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename VertexCondition, typename Function, typename T >
void
forVerticesIf( Graph& graph,
               IndexBegin begin,
               IndexEnd end,
               VertexCondition&& vertexCondition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::TraversingOperations< typename Graph::ViewType >::forVerticesIf( graph_view,
                                                                            begin,
                                                                            end,
                                                                            std::forward< VertexCondition >( vertexCondition ),
                                                                            std::forward< Function >( function ),
                                                                            launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename VertexCondition, typename Function, typename T >
void
forVerticesIf( const Graph& graph,
               IndexBegin begin,
               IndexEnd end,
               VertexCondition&& vertexCondition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Graph::ConstViewType >::forVerticesIf(
      graph.getConstView(),
      begin,
      end,
      std::forward< VertexCondition >( vertexCondition ),
      std::forward< Function >( function ),
      launchConfig );
}

template< typename Graph, typename VertexCondition, typename Function >
void
forAllVerticesIf( Graph& graph,
                  VertexCondition&& vertexCondition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
{
   forVerticesIf( graph,
                  (typename Graph::IndexType) 0,
                  graph.getVertexCount(),
                  std::forward< VertexCondition >( vertexCondition ),
                  std::forward< Function >( function ),
                  launchConfig );
}

template< typename Graph, typename VertexCondition, typename Function >
void
forAllVerticesIf( const Graph& graph,
                  VertexCondition&& vertexCondition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
{
   forVerticesIf( graph,
                  (typename Graph::IndexType) 0,
                  graph.getVertexCount(),
                  std::forward< VertexCondition >( vertexCondition ),
                  std::forward< Function >( function ),
                  launchConfig );
}

}  //namespace TNL::Graphs
