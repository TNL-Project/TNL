// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Matrices/reduce.h>
#include "../SubGraph.h"
#include "ReductionOperations.h"

namespace TNL::Graphs::detail {

/**
 * \brief Specialization of ReductionOperations for SubGraph.
 *
 * Applies the vertex filter (skip edges touching inactive vertices) and the
 * edge filter (skip filtered edges) transparently during reduction, so that
 * \ref reduceAllVertices, \ref reduceVertices, \ref reduceVerticesIf, etc.
 * work with SubGraph the same way they work with Graph.
 *
 * Inactive vertices produce the identity value because every fetched element
 * in their row returns identity. For \c If variants, the vertex filter is
 * combined with the user-supplied condition so inactive rows are skipped
 * entirely.
 */
template< typename Graph_, typename VertexFilter, typename EdgeFilter >
struct ReductionOperations< SubGraph< Graph_, VertexFilter, EdgeFilter > >
{
   using SubGraphType = SubGraph< Graph_, VertexFilter, EdgeFilter >;
   using GraphType = typename SubGraphType::GraphType;
   using ValueType = typename SubGraphType::ValueType;
   using DeviceType = typename SubGraphType::DeviceType;
   using IndexType = typename SubGraphType::IndexType;
   using AdjacencyMatrixView = typename SubGraphType::ConstAdjacencyMatrixView;

   // -----------------------------------------------------------------
   // reduceVertices (range)
   // -----------------------------------------------------------------

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVertices(
      const SubGraphType& graph,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      const auto& edgeFilter = graph.getEdgeFilter();
      auto wrappedFetch =
         [ = ] __cuda_callable__( IndexType row, IndexType column, const ValueType& value ) mutable -> FetchValue
      {
         if( vertexFilter( row ) && vertexFilter( column ) && edgeFilter( row, column, value ) )
            return fetch( row, column, value );
         return identity;
      };
      Matrices::reduceRows(
         graph.getAdjacencyMatrixView(), begin, end, wrappedFetch, reduction, store, identity, launchConfig );
   }

   // -----------------------------------------------------------------
   // reduceVertices (array)
   // -----------------------------------------------------------------

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVertices(
      const SubGraphType& graph,
      const Array& vertexIndices,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      const auto& edgeFilter = graph.getEdgeFilter();
      auto wrappedFetch =
         [ = ] __cuda_callable__( IndexType row, IndexType column, const ValueType& value ) mutable -> FetchValue
      {
         if( vertexFilter( row ) && vertexFilter( column ) && edgeFilter( row, column, value ) )
            return fetch( row, column, value );
         return identity;
      };
      Matrices::reduceRows(
         graph.getAdjacencyMatrixView(), vertexIndices, wrappedFetch, reduction, store, identity, launchConfig );
   }

   // -----------------------------------------------------------------
   // reduceVerticesIf (range + condition)
   // -----------------------------------------------------------------

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceVerticesIf(
      const SubGraphType& graph,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      const auto& edgeFilter = graph.getEdgeFilter();
      auto combinedCondition = [ = ] __cuda_callable__( IndexType row ) mutable -> bool
      {
         return condition( row ) && vertexFilter( row );
      };
      auto wrappedFetch =
         [ = ] __cuda_callable__( IndexType row, IndexType column, const ValueType& value ) mutable -> FetchValue
      {
         if( vertexFilter( column ) && edgeFilter( row, column, value ) )
            return fetch( row, column, value );
         return identity;
      };
      return Matrices::reduceRowsIf(
         graph.getAdjacencyMatrixView(),
         begin,
         end,
         combinedCondition,
         wrappedFetch,
         reduction,
         store,
         identity,
         launchConfig );
   }

   // -----------------------------------------------------------------
   // reduceVerticesWithArgument (range)
   // -----------------------------------------------------------------

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVerticesWithArgument(
      const SubGraphType& graph,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      const auto& edgeFilter = graph.getEdgeFilter();
      auto wrappedFetch =
         [ = ] __cuda_callable__( IndexType row, IndexType column, const ValueType& value ) mutable -> FetchValue
      {
         if( vertexFilter( row ) && vertexFilter( column ) && edgeFilter( row, column, value ) )
            return fetch( row, column, value );
         return identity;
      };
      Matrices::reduceRowsWithArgument(
         graph.getAdjacencyMatrixView(), begin, end, wrappedFetch, reduction, store, identity, launchConfig );
   }

   // -----------------------------------------------------------------
   // reduceVerticesWithArgument (array)
   // -----------------------------------------------------------------

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVerticesWithArgument(
      const SubGraphType& graph,
      const Array& vertexIndices,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      const auto& edgeFilter = graph.getEdgeFilter();
      auto wrappedFetch =
         [ = ] __cuda_callable__( IndexType row, IndexType column, const ValueType& value ) mutable -> FetchValue
      {
         if( vertexFilter( row ) && vertexFilter( column ) && edgeFilter( row, column, value ) )
            return fetch( row, column, value );
         return identity;
      };
      Matrices::reduceRowsWithArgument(
         graph.getAdjacencyMatrixView(), vertexIndices, wrappedFetch, reduction, store, identity, launchConfig );
   }

   // -----------------------------------------------------------------
   // reduceVerticesWithArgumentIf (range + condition)
   // -----------------------------------------------------------------

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceVerticesWithArgumentIf(
      const SubGraphType& graph,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      const auto& vertexFilter = graph.getVertexFilter();
      const auto& edgeFilter = graph.getEdgeFilter();
      auto combinedCondition = [ = ] __cuda_callable__( IndexType row ) mutable -> bool
      {
         return condition( row ) && vertexFilter( row );
      };
      auto wrappedFetch =
         [ = ] __cuda_callable__( IndexType row, IndexType column, const ValueType& value ) mutable -> FetchValue
      {
         if( vertexFilter( column ) && edgeFilter( row, column, value ) )
            return fetch( row, column, value );
         return identity;
      };
      return Matrices::reduceRowsWithArgumentIf(
         graph.getAdjacencyMatrixView(),
         begin,
         end,
         combinedCondition,
         wrappedFetch,
         reduction,
         store,
         identity,
         launchConfig );
   }
};

}  // namespace TNL::Graphs::detail
