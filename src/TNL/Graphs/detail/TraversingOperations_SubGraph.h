// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/traverse.h>
#include "../SubGraph.h"
#include "TraversingOperations.h"

namespace TNL::Graphs::detail {

/**
 * \brief Specialization of TraversingOperations for SubGraph.
 *
 * Applies the vertex filter (skip inactive source vertices) and the edge
 * filter (skip filtered edges) transparently during traversal, so that
 * \ref forAllEdges, \ref forEdges, \ref forEdgesIf, etc. work with SubGraph
 * the same way they work with Graph — no algorithm-level branching needed.
 *
 * SubGraph IS its own ViewType / ConstViewType (see SubGraph.h), so
 * traverse.hpp dispatches here when the graph parameter is a SubGraph.
 */
template< typename Graph_, typename VertexFilter, typename EdgeFilter >
struct TraversingOperations< SubGraph< Graph_, VertexFilter, EdgeFilter > >
{
   using SubGraphType = SubGraph< Graph_, VertexFilter, EdgeFilter >;
   using GraphType = typename SubGraphType::GraphType;
   using ValueType = typename SubGraphType::ValueType;
   using DeviceType = typename SubGraphType::DeviceType;
   using IndexType = typename SubGraphType::IndexType;
   using AdjacencyMatrixView = typename SubGraphType::ConstAdjacencyMatrixView;
   using RowViewType = typename AdjacencyMatrixView::RowView;
   using ConstRowViewType = typename AdjacencyMatrixView::ConstRowView;
   using VertexView = typename SubGraphType::VertexView;
   using ConstVertexView = typename SubGraphType::ConstVertexView;

   // -----------------------------------------------------------------
   // forEdges (range)
   // -----------------------------------------------------------------

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forEdges(
      const SubGraphType& graph,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      const auto& edgeFilter = graph.getEdgeFilter();
      auto wrapped =
         [ = ] __cuda_callable__( IndexType row, IndexType localIdx, IndexType column, const ValueType& value ) mutable
      {
         if( vertexFilter( row ) && edgeFilter( row, column, value ) )
            function( row, localIdx, column, value );
      };
      Matrices::forElements( graph.getAdjacencyMatrixView(), begin, end, wrapped, launchConfig );
   }

   // -----------------------------------------------------------------
   // forEdges (row indexes)
   // -----------------------------------------------------------------

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forEdges(
      const SubGraphType& graph,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      const auto& edgeFilter = graph.getEdgeFilter();
      auto wrapped =
         [ = ] __cuda_callable__( IndexType row, IndexType localIdx, IndexType column, const ValueType& value ) mutable
      {
         if( vertexFilter( row ) && edgeFilter( row, column, value ) )
            function( row, localIdx, column, value );
      };
      Matrices::forElements( graph.getAdjacencyMatrixView(), rowIndexes.getConstView( begin, end ), wrapped, launchConfig );
   }

   // -----------------------------------------------------------------
   // forEdgesIf (range + condition)
   // -----------------------------------------------------------------

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forEdgesIf(
      const SubGraphType& graph,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      const auto& edgeFilter = graph.getEdgeFilter();
      auto combinedCondition = [ = ] __cuda_callable__( IndexType row ) mutable -> bool
      {
         return condition( row ) && vertexFilter( row );
      };
      auto wrapped =
         [ = ] __cuda_callable__( IndexType row, IndexType localIdx, IndexType column, const ValueType& value ) mutable
      {
         if( edgeFilter( row, column, value ) )
            function( row, localIdx, column, value );
      };
      Matrices::forElementsIf( graph.getAdjacencyMatrixView(), begin, end, combinedCondition, wrapped, launchConfig );
   }

   // -----------------------------------------------------------------
   // forVertices (range)
   // -----------------------------------------------------------------

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forVertices(
      const SubGraphType& graph,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      auto wrapped = [ = ] __cuda_callable__( const ConstRowViewType& rowView ) mutable
      {
         if( vertexFilter( rowView.getRowIndex() ) )
            function( ConstVertexView( rowView ) );
      };
      Matrices::forRows( graph.getAdjacencyMatrixView(), begin, end, wrapped, launchConfig );
   }

   // -----------------------------------------------------------------
   // forVertices (row indexes)
   // -----------------------------------------------------------------

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forVertices(
      const SubGraphType& graph,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Function&& function,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      auto wrapped = [ = ] __cuda_callable__( const ConstRowViewType& rowView ) mutable
      {
         if( vertexFilter( rowView.getRowIndex() ) )
            function( ConstVertexView( rowView ) );
      };
      Matrices::forRows( graph.getAdjacencyMatrixView(), rowIndexes.getConstView( begin, end ), wrapped, launchConfig );
   }

   // -----------------------------------------------------------------
   // forVerticesIf (range + condition)
   // -----------------------------------------------------------------

   template< typename IndexBegin, typename IndexEnd, typename VertexCondition, typename Function >
   static void
   forVerticesIf(
      const SubGraphType& graph,
      IndexBegin begin,
      IndexEnd end,
      VertexCondition&& vertexCondition,
      Function&& function,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      auto combinedCondition = [ = ] __cuda_callable__( IndexType row ) mutable -> bool
      {
         return vertexCondition( row ) && vertexFilter( row );
      };
      auto wrapped = [ = ] __cuda_callable__( const ConstRowViewType& rowView ) mutable
      {
         function( ConstVertexView( rowView ) );
      };
      Matrices::forRowsIf( graph.getAdjacencyMatrixView(), begin, end, combinedCondition, wrapped, launchConfig );
   }
};

}  // namespace TNL::Graphs::detail
