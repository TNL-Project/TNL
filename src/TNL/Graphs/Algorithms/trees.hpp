// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>
#include <stdexcept>
#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/find.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Functional.h>
#include <TNL/Matrices/MatrixBase.h>

#include "details/activeVertices.hpp"
#include "trees.h"

namespace TNL::Graphs::Algorithms {

enum class TreeType : std::uint8_t
{
   Tree,
   Forest
};

namespace detail {

template< typename EdgePredicate, typename Graph >
struct IsTreeEdgePredicate
: std::bool_constant<
     std::
        is_invocable_r_v< bool, EdgePredicate, typename Graph::IndexType, typename Graph::IndexType, typename Graph::ValueType > >
{};

template< typename EdgePredicate, typename Graph >
constexpr bool isTreeEdgePredicate_v = IsTreeEdgePredicate< EdgePredicate, Graph >::value;

template< typename Graph, typename IsActive, typename EdgePredicate >
typename Graph::IndexType
countActiveEdges( const Graph& graph, IsActive&& isActive, EdgePredicate&& edgePredicate )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   const IndexType n = graph.getVertexCount();
   const auto& matrix = graph.getAdjacencyMatrix();
   constexpr bool isUndirected = Graph::isUndirected();

   if constexpr( std::is_same_v< DeviceType, Devices::Sequential > ) {
      IndexType edgeCount = 0;
      for( IndexType rowIdx = 0; rowIdx < n; rowIdx++ ) {
         if( ! isActive( rowIdx ) )
            continue;
         const auto row = matrix.getRow( rowIdx );
         for( IndexType i = 0; i < row.getSize(); i++ ) {
            const auto col = row.getColumnIndex( i );
            if( col == Matrices::paddingIndex< IndexType > )
               continue;
            if( ! isActive( col ) )
               continue;
            const ValueType weight = row.getValue( i );
            if( ! edgePredicate( rowIdx, col, weight ) )
               continue;
            if constexpr( isUndirected ) {
               if( col <= rowIdx )
                  continue;
            }
            edgeCount++;
         }
      }
      if constexpr( isUndirected ) {
         for( IndexType rowIdx = 0; rowIdx < n; rowIdx++ ) {
            if( ! isActive( rowIdx ) )
               continue;
            const auto row = matrix.getRow( rowIdx );
            for( IndexType i = 0; i < row.getSize(); i++ ) {
               const auto col = row.getColumnIndex( i );
               if( col == Matrices::paddingIndex< IndexType > || col != rowIdx )
                  continue;
               if( ! edgePredicate( rowIdx, col, row.getValue( i ) ) )
                  continue;
               edgeCount++;
            }
         }
      }
      return edgeCount;
   }
   else {
      Containers::Vector< IndexType, DeviceType, IndexType > edgeCounts( n, 0 );
      auto edgeCounts_view = edgeCounts.getView();
      auto active_view = std::forward< IsActive >( isActive );
      auto edge_pred = std::forward< EdgePredicate >( edgePredicate );

      auto fetch_edge =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
      {
         if( columnIdx == Matrices::paddingIndex< IndexType > )
            return 0;
         if( ! active_view( rowIdx ) || ! active_view( columnIdx ) )
            return 0;
         if( ! edge_pred( rowIdx, columnIdx, value ) )
            return 0;
         if constexpr( isUndirected ) {
            if( columnIdx <= rowIdx )
               return 0;
         }
         return 1;
      };
      auto keep = [ = ] __cuda_callable__( IndexType rowIdx, const IndexType value ) mutable
      {
         edgeCounts_view[ rowIdx ] = value;
      };
      matrix.reduceAllRows( fetch_edge, TNL::Plus{}, keep, (IndexType) 0 );

      IndexType total = TNL::Algorithms::reduce< DeviceType >(
         0,
         n,
         [ = ] __cuda_callable__( IndexType idx ) -> IndexType
         {
            return edgeCounts_view[ idx ];
         },
         TNL::Plus{} );

      if constexpr( isUndirected ) {
         Containers::Vector< IndexType, DeviceType, IndexType > diagonalCounts( n, 0 );
         auto diag_view = diagonalCounts.getView();
         auto diag_fetch =
            [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
         {
            if( columnIdx != rowIdx || columnIdx == Matrices::paddingIndex< IndexType > )
               return 0;
            if( ! active_view( rowIdx ) )
               return 0;
            if( ! edge_pred( rowIdx, columnIdx, value ) )
               return 0;
            return 1;
         };
         auto diag_keep = [ = ] __cuda_callable__( IndexType rowIdx, const IndexType value ) mutable
         {
            diag_view[ rowIdx ] = value;
         };
         matrix.reduceAllRows( diag_fetch, TNL::Plus{}, diag_keep, (IndexType) 0 );
         total += TNL::Algorithms::reduce< DeviceType >(
            0,
            n,
            [ = ] __cuda_callable__( IndexType idx ) -> IndexType
            {
               return diag_view[ idx ];
            },
            TNL::Plus{} );
      }
      return total;
   }
}

template< typename Graph, typename IsActive >
typename Graph::IndexType
countActiveVertices( const Graph& graph, IsActive&& isActive )
{
   using IndexType = typename Graph::IndexType;
   using DeviceType = typename Graph::DeviceType;
   const IndexType n = graph.getVertexCount();

   return TNL::Algorithms::reduce< DeviceType >(
      0,
      n,
      [ pred = std::forward< IsActive >( isActive ) ] __cuda_callable__( IndexType idx ) -> IndexType
      {
         return pred( idx ) ? 1 : 0;
      },
      TNL::Plus{} );
}

}  // namespace detail

template< typename Vector, typename Index = typename Vector::IndexType >
bool
visitNeighbour( const Index current, const Index neighbor, Vector& visited, Vector& parents, std::queue< Index >& q )
{
   if( neighbor == parents[ current ] )
      return true;
   if( visited[ neighbor ] )
      return false;
   parents[ neighbor ] = current;
   visited[ neighbor ] = 1;
   q.push( neighbor );
   return true;
}

template< typename Graph, typename Vector, typename ActivePredicate, typename EdgePredicate >
bool
isTree_impl(
   const Graph& graph,
   const Vector& roots,
   TreeType treeType,
   ActivePredicate&& isActive,
   EdgePredicate&& edgePredicate )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
   using AdjacencyMatrixType = typename Graph::AdjacencyMatrixType;

   const IndexType n = graph.getVertexCount();

   if( treeType == TreeType::Tree ) {
      const IndexType nActive = detail::countActiveVertices( graph, isActive );
      const IndexType activeEdgeCount = detail::countActiveEdges( graph, isActive, edgePredicate );
      if( activeEdgeCount != nActive - 1 )
         return false;
   }

   IndexVectorType visited( n, 0 );
   IndexVectorType visited_old( n, -1 );
   IndexVectorType parents( n, 0 );
   IndexType start = 0;
   IndexType rootsIdx = 0;
   if( ! roots.empty() )
      start = roots.getElement( rootsIdx++ );
   else {
      for( IndexType i = 0; i < n; i++ ) {
         if( isActive( i ) ) {
            start = i;
            break;
         }
      }
   }
   while( true ) {
      if( ! isActive( start ) )
         return false;
      visited.setElement( start, 1 );
      if constexpr( std::is_same_v< DeviceType, Devices::Sequential > ) {
         std::queue< IndexType > q;
         q.push( start );
         while( ! q.empty() ) {
            IndexType current = q.front();
            q.pop();
            const auto row = graph.getAdjacencyMatrix().getRow( current );
            for( IndexType i = 0; i < row.getSize(); i++ ) {
               const auto neighbor = row.getColumnIndex( i );
               if( neighbor == Matrices::paddingIndex< IndexType > )
                  continue;
               if( ! isActive( neighbor ) )
                  continue;
               const ValueType weight = row.getValue( i );
               if( ! edgePredicate( current, neighbor, weight ) )
                  continue;
               if( ! visitNeighbour( current, neighbor, visited, parents, q ) )
                  return false;
            }
            if constexpr( AdjacencyMatrixType::isSymmetric() ) {
               for( IndexType rowIdx = 0; rowIdx < graph.getVertexCount(); rowIdx++ ) {
                  if( rowIdx == current )
                     continue;
                  if( ! isActive( rowIdx ) )
                     continue;
                  auto row2 = graph.getAdjacencyMatrix().getRow( rowIdx );
                  for( IndexType i = 0; i < row2.getSize(); i++ ) {
                     const auto col = row2.getColumnIndex( i );
                     if( col == Matrices::paddingIndex< IndexType > || col != current )
                        continue;
                     const ValueType weight = row2.getValue( i );
                     if( ! edgePredicate( rowIdx, current, weight ) )
                        continue;
                     if( ! visitNeighbour( current, rowIdx, visited, parents, q ) )
                        return false;
                  }
               }
            }
         }
      }
      else {
         auto isActiveCopy = isActive;
         auto edgePredicateCopy = edgePredicate;
         while( visited_old != visited ) {
            visited_old = visited;
            auto visited_view = visited.getView();
            auto visited_old_view = visited_old.getView();
            auto symmetric_fetch =
               [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
            {
               if( columnIdx == Matrices::paddingIndex< IndexType > )
                  return 0;
               if( ! isActiveCopy( columnIdx ) || ! isActiveCopy( rowIdx ) )
                  return 0;
               if( ! edgePredicateCopy( rowIdx, columnIdx, value ) )
                  return 0;
               if( ! visited_old_view[ columnIdx ] )
                  TNL::Algorithms::AtomicOperations< DeviceType >::add( visited_view[ columnIdx ], visited_old_view[ rowIdx ] );
               if( visited_old_view[ rowIdx ] )
                  return 0;
               return visited_old_view[ columnIdx ] != 0;
            };
            auto fetch =
               [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
            {
               if( columnIdx == Matrices::paddingIndex< IndexType > )
                  return 0;
               if( ! isActiveCopy( columnIdx ) || ! isActiveCopy( rowIdx ) )
                  return 0;
               if( ! edgePredicateCopy( rowIdx, columnIdx, value ) )
                  return 0;
               if( visited_old_view[ rowIdx ] )
                  return 0;
               return visited_old_view[ columnIdx ] != 0;
            };
            auto keep = [ = ] __cuda_callable__( IndexType rowIdx, const IndexType value ) mutable
            {
               visited_view[ rowIdx ] = visited_view[ rowIdx ] + value;
            };
            if constexpr( AdjacencyMatrixType::isSymmetric() )
               graph.getAdjacencyMatrix().reduceAllRows( symmetric_fetch, TNL::Plus{}, keep, (IndexType) 0 );
            else
               graph.getAdjacencyMatrix().reduceAllRows( fetch, TNL::Plus{}, keep, (IndexType) 0 );

            bool anyExceeded = false;
            for( IndexType i = 0; i < n; i++ ) {
               if( isActive( i ) && visited[ i ] > 1 ) {
                  anyExceeded = true;
                  break;
               }
            }
            if( anyExceeded )
               return false;
            bool allVisited = true;
            for( IndexType i = 0; i < n; i++ ) {
               if( isActive( i ) && visited[ i ] != 1 ) {
                  allVisited = false;
                  break;
               }
            }
            if( allVisited )
               return true;
         }
      }
      bool allVisitedSeq = true;
      for( IndexType i = 0; i < n; i++ ) {
         if( isActive( i ) && visited[ i ] != 1 ) {
            allVisitedSeq = false;
            break;
         }
      }
      if( allVisitedSeq )
         return true;
      if( treeType == TreeType::Tree )
         return false;
      if( ! roots.empty() ) {
         if( rootsIdx < roots.getSize() )
            start = roots.getElement( rootsIdx++ );
         else
            return false;
      }
      else {
         for( IndexType i = 0; i < n; i++ ) {
            if( isActive( i ) && visited[ i ] == 0 ) {
               start = i;
               goto found_next;
            }
         }
         return true;
found_next:;
      }
   }
}

// isTree overloads

template< typename Graph >
bool
isTree( const Graph& graph, typename Graph::IndexType start )
{
   using IndexType = typename Graph::IndexType;

   Containers::Vector< IndexType > roots( 1, start );
   return isTree_impl(
      graph,
      roots,
      TreeType::Tree,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      } );
}

template< typename Graph, typename EdgePredicate, typename >
bool
isTree( const Graph& graph, typename Graph::IndexType start, EdgePredicate&& edgePredicate )
{
   using IndexType = typename Graph::IndexType;

   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isTree edge predicate must return bool and accept (source, target, weight)." );

   Containers::Vector< IndexType > roots( 1, start );
   return isTree_impl(
      graph,
      roots,
      TreeType::Tree,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      std::forward< EdgePredicate >( edgePredicate ) );
}

template< typename Graph, typename VertexIndexes, typename >
bool
isTree( const Graph& graph, typename Graph::IndexType start, const VertexIndexes& vertexIndexes )
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

   Containers::Vector< IndexType > roots( 1, start );
   return isTree_impl(
      graph,
      roots,
      TreeType::Tree,
      isActive,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      } );
}

template< typename Graph, typename VertexIndexes, typename EdgePredicate, typename >
bool
isTree( const Graph& graph, typename Graph::IndexType start, const VertexIndexes& vertexIndexes, EdgePredicate&& edgePredicate )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isTree edge predicate must return bool and accept (source, target, weight)." );

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   const auto isActive = [ = ] __cuda_callable__( IndexType vertex )
   {
      return static_cast< bool >( activeVerticesView[ vertex ] );
   };

   Containers::Vector< IndexType > roots( 1, start );
   return isTree_impl( graph, roots, TreeType::Tree, isActive, std::forward< EdgePredicate >( edgePredicate ) );
}

template< typename Graph, typename VertexPredicate >
bool
isTreeIf( const Graph& graph, typename Graph::IndexType start, VertexPredicate&& vertexPredicate )
{
   using IndexType = typename Graph::IndexType;

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   Containers::Vector< IndexType > roots( 1, start );
   return isTree_impl(
      graph,
      roots,
      TreeType::Tree,
      predicate,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      } );
}

template< typename Graph, typename VertexPredicate, typename EdgePredicate >
bool
isTreeIf(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate )
{
   using IndexType = typename Graph::IndexType;

   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isTree edge predicate must return bool and accept (source, target, weight)." );

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   Containers::Vector< IndexType > roots( 1, start );
   return isTree_impl( graph, roots, TreeType::Tree, predicate, std::forward< EdgePredicate >( edgePredicate ) );
}

// isForest overloads (auto roots)

template< typename Graph >
bool
isForest( const Graph& graph )
{
   using IndexType = typename Graph::IndexType;

   Containers::Vector< IndexType > roots;
   return isTree_impl(
      graph,
      roots,
      TreeType::Forest,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      } );
}

template< typename Graph, typename EdgePredicate, typename >
bool
isForest( const Graph& graph, EdgePredicate&& edgePredicate )
{
   using IndexType = typename Graph::IndexType;

   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isForest edge predicate must return bool and accept (source, target, weight)." );

   Containers::Vector< IndexType > roots;
   return isTree_impl(
      graph,
      roots,
      TreeType::Forest,
      [] __cuda_callable__( IndexType )
      {
         return true;
      },
      std::forward< EdgePredicate >( edgePredicate ) );
}

template< typename Graph, typename VertexIndexes, typename >
bool
isForest( const Graph& graph, const VertexIndexes& vertexIndexes )
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

   Containers::Vector< IndexType > roots;
   return isTree_impl(
      graph,
      roots,
      TreeType::Forest,
      isActive,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      } );
}

template< typename Graph, typename VertexIndexes, typename EdgePredicate, typename >
bool
isForest( const Graph& graph, const VertexIndexes& vertexIndexes, EdgePredicate&& edgePredicate )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isForest edge predicate must return bool and accept (source, target, weight)." );

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   const auto isActive = [ = ] __cuda_callable__( IndexType vertex )
   {
      return static_cast< bool >( activeVerticesView[ vertex ] );
   };

   Containers::Vector< IndexType > roots;
   return isTree_impl( graph, roots, TreeType::Forest, isActive, std::forward< EdgePredicate >( edgePredicate ) );
}

template< typename Graph, typename VertexPredicate >
bool
isForestIf( const Graph& graph, VertexPredicate&& vertexPredicate )
{
   using IndexType = typename Graph::IndexType;

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   Containers::Vector< IndexType > roots;
   return isTree_impl(
      graph,
      roots,
      TreeType::Forest,
      predicate,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      } );
}

template< typename Graph, typename VertexPredicate, typename EdgePredicate >
bool
isForestIf( const Graph& graph, VertexPredicate&& vertexPredicate, EdgePredicate&& edgePredicate )
{
   using IndexType = typename Graph::IndexType;

   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isForest edge predicate must return bool and accept (source, target, weight)." );

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   Containers::Vector< IndexType > roots;
   return isTree_impl( graph, roots, TreeType::Forest, predicate, std::forward< EdgePredicate >( edgePredicate ) );
}

// isForestWithRoots overloads (explicit roots)

template< typename Graph, typename Vector >
bool
isForestWithRoots( const Graph& graph, const Vector& roots )
{
   return isTree_impl(
      graph,
      roots,
      TreeType::Forest,
      [] __cuda_callable__( typename Graph::IndexType )
      {
         return true;
      },
      [] __cuda_callable__( typename Graph::IndexType, typename Graph::IndexType, auto )
      {
         return true;
      } );
}

template< typename Graph, typename EdgePredicate, typename Vector, typename >
bool
isForestWithRoots( const Graph& graph, EdgePredicate&& edgePredicate, const Vector& roots )
{
   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isForestWithRoots edge predicate must return bool and accept (source, target, weight)." );

   return isTree_impl(
      graph,
      roots,
      TreeType::Forest,
      [] __cuda_callable__( typename Graph::IndexType )
      {
         return true;
      },
      std::forward< EdgePredicate >( edgePredicate ) );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
bool
isForestWithRoots( const Graph& graph, const VertexIndexes& vertexIndexes, const Vector& roots )
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

   return isTree_impl(
      graph,
      roots,
      TreeType::Forest,
      isActive,
      [] __cuda_callable__( IndexType, IndexType, auto )
      {
         return true;
      } );
}

template< typename Graph, typename VertexIndexes, typename EdgePredicate, typename Vector, typename >
bool
isForestWithRoots( const Graph& graph, const VertexIndexes& vertexIndexes, EdgePredicate&& edgePredicate, const Vector& roots )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isForestWithRoots edge predicate must return bool and accept (source, target, weight)." );

   IndexVector activeVertices;
   detail::activateIndexedVertices( graph, vertexIndexes, activeVertices );
   const auto activeVerticesView = activeVertices.getConstView();
   const auto isActive = [ = ] __cuda_callable__( IndexType vertex )
   {
      return static_cast< bool >( activeVerticesView[ vertex ] );
   };

   return isTree_impl( graph, roots, TreeType::Forest, isActive, std::forward< EdgePredicate >( edgePredicate ) );
}

template< typename Graph, typename VertexPredicate, typename Vector >
bool
isForestWithRootsIf( const Graph& graph, VertexPredicate&& vertexPredicate, const Vector& roots )
{
   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   return isTree_impl(
      graph,
      roots,
      TreeType::Forest,
      predicate,
      [] __cuda_callable__( typename Graph::IndexType, typename Graph::IndexType, auto )
      {
         return true;
      } );
}

template< typename Graph, typename VertexPredicate, typename EdgePredicate, typename Vector >
bool
isForestWithRootsIf( const Graph& graph, VertexPredicate&& vertexPredicate, EdgePredicate&& edgePredicate, const Vector& roots )
{
   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isForestWithRootsIf edge predicate must return bool and accept (source, target, weight)." );

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   return isTree_impl( graph, roots, TreeType::Forest, predicate, std::forward< EdgePredicate >( edgePredicate ) );
}

}  // namespace TNL::Graphs::Algorithms
