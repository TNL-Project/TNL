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
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
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
countActiveEdges(
   const Graph& graph,
   IsActive&& isActive,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using AdjacencyMatrixType = typename Graph::AdjacencyMatrixType;
   const IndexType n = graph.getVertexCount();
   const auto& matrix = graph.getAdjacencyMatrix();
   constexpr bool isUndirected = Graph::isUndirected();
   constexpr bool isSymmetric = AdjacencyMatrixType::isSymmetric();

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
            if constexpr( isUndirected && ! isSymmetric ) {
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
      auto edgeCountsView = edgeCounts.getView();
      auto activeView = std::forward< IsActive >( isActive );
      auto edgePredicateView = std::forward< EdgePredicate >( edgePredicate );

      auto fetch_edge =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
      {
         if( columnIdx == Matrices::paddingIndex< IndexType > )
            return 0;
         if( ! activeView( rowIdx ) || ! activeView( columnIdx ) )
            return 0;
         if( ! edgePredicateView( rowIdx, columnIdx, value ) )
            return 0;
         if constexpr( isUndirected && ! isSymmetric ) {
            if( columnIdx <= rowIdx )
               return 0;
         }
         return 1;
      };
      auto keep = [ = ] __cuda_callable__( IndexType rowIdx, const IndexType value ) mutable
      {
         edgeCountsView[ rowIdx ] = value;
      };
      matrix.reduceAllRows( fetch_edge, Plus{}, keep, (IndexType) 0, launchConfig );

      IndexType total = TNL::Algorithms::reduce< DeviceType >(
         0,
         n,
         [ = ] __cuda_callable__( IndexType idx ) -> IndexType
         {
            return edgeCountsView[ idx ];
         },
         Plus{} );

      if constexpr( isUndirected ) {
         Containers::Vector< IndexType, DeviceType, IndexType > diagonalCounts( n, 0 );
         auto diagView = diagonalCounts.getView();
         auto diag_fetch =
            [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
         {
            if( columnIdx != rowIdx || columnIdx == Matrices::paddingIndex< IndexType > )
               return 0;
            if( ! activeView( rowIdx ) )
               return 0;
            if( ! edgePredicateView( rowIdx, columnIdx, value ) )
               return 0;
            return 1;
         };
         auto diag_keep = [ = ] __cuda_callable__( IndexType rowIdx, const IndexType value ) mutable
         {
            diagView[ rowIdx ] = value;
         };
         matrix.reduceAllRows( diag_fetch, Plus{}, diag_keep, (IndexType) 0, launchConfig );
         total += TNL::Algorithms::reduce< DeviceType >(
            0,
            n,
            [ = ] __cuda_callable__( IndexType idx ) -> IndexType
            {
               return diagView[ idx ];
            },
            Plus{} );
      }
      return total;
   }
}

template< typename Graph, typename IsActive >
typename Graph::IndexType
countActiveVertices( const Graph& graph, IsActive&& isActive, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      Plus{} );
}

}  // namespace detail

// Visit a neighbor during sequential BFS tree-checking.
// Returns true if the edge (current -> neighbor) is a valid tree edge
// (neighbor was unvisited or is the parent of current).
// Returns false if a cycle/cross-edge is detected (neighbor already
// visited and is not the parent of current).
template< typename Vector, typename Index = typename Vector::IndexType >
bool
visitNeighbor( const Index current, const Index neighbor, Vector& visited, Vector& parents, std::queue< Index >& q )
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
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
   using AdjacencyMatrixType = typename Graph::AdjacencyMatrixType;

   const IndexType n = graph.getVertexCount();

   if( treeType == TreeType::Tree ) {
      // A tree on n vertices has exactly n-1 edges.  Guard against the
      // unsigned underflow when nActive == 0 (an empty graph is trivially
      // a tree — zero vertices, zero edges).
      const IndexType nActive = detail::countActiveVertices( graph, isActive, launchConfig );
      if( nActive == 0 )
         return true;
      const IndexType activeEdgeCount = detail::countActiveEdges( graph, isActive, edgePredicate, launchConfig );
      if( activeEdgeCount != nActive - 1 )
         return false;
   }

   IndexVectorType visited( n, 0 );
   IndexVectorType visited_old( n, -1 );
   IndexVectorType parents( n, -1 );
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
   // BFS from the start vertex.  If TreeType::Tree, we return false as
   // soon as a cycle/cross-edge is found or not all active vertices are
   // reached.  For TreeType::Forest we restart from the next unvisited
   // active vertex (or the next explicit root) until all are covered.
   while( true ) {
      if( ! isActive( start ) )
         return false;
      visited.setElement( start, 1 );
      parents.setElement( start, start );
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
               if( ! visitNeighbor( current, neighbor, visited, parents, q ) )
                  return false;
            }
            if constexpr( AdjacencyMatrixType::isSymmetric() ) {
               // Symmetric matrices store only the lower triangle, so
               // getRow(current) misses neighbors j > current.  Scan all
               // rows to find edges pointing TO current (rowIdx -> current),
               // which correspond to the missing upper-triangle entries.
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
                     if( ! visitNeighbor( current, rowIdx, visited, parents, q ) )
                        return false;
                  }
               }
            }
         }
      }
      else {
         auto isActiveCopy = isActive;
         auto edgePredicateCopy = edgePredicate;
         // Iterative BFS: each round propagates "visited" from the current
         // frontier to its neighbors via reduceAllRows.  If a vertex is
         // reached more than once per round (visited > 1), it indicates a
         // cycle or cross-edge, so the graph is not a tree/forest.
         while( visited_old != visited ) {
            visited_old = visited;
            auto visitedView = visited.getView();
            auto visitedOldView = visited_old.getView();
            // For symmetric matrices (lower-triangle storage), we must also
            // propagate in the reverse direction: when rowIdx is visited and
            // columnIdx is not, mark columnIdx.  This extra atomic add
            // compensates for the missing upper-triangle entries.
            auto symmetric_fetch =
               [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
            {
               if( columnIdx == Matrices::paddingIndex< IndexType > )
                  return 0;
               if( ! isActiveCopy( columnIdx ) || ! isActiveCopy( rowIdx ) )
                  return 0;
               if( ! edgePredicateCopy( rowIdx, columnIdx, value ) )
                  return 0;
               if( ! visitedOldView[ columnIdx ] )
                  TNL::Algorithms::AtomicOperations< DeviceType >::add( visitedView[ columnIdx ], visitedOldView[ rowIdx ] );
               if( visitedOldView[ rowIdx ] )
                  return 0;
               return visitedOldView[ columnIdx ] != 0;
            };
            // For non-symmetric matrices, both directions are stored
            // explicitly, so forward propagation suffices.
            auto fetch =
               [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
            {
               if( columnIdx == Matrices::paddingIndex< IndexType > )
                  return 0;
               if( ! isActiveCopy( columnIdx ) || ! isActiveCopy( rowIdx ) )
                  return 0;
               if( ! edgePredicateCopy( rowIdx, columnIdx, value ) )
                  return 0;
               if( visitedOldView[ rowIdx ] )
                  return 0;
               return visitedOldView[ columnIdx ] != 0;
            };
            auto keep = [ = ] __cuda_callable__( IndexType rowIdx, const IndexType value ) mutable
            {
               visitedView[ rowIdx ] = visitedView[ rowIdx ] + value;
            };
            if constexpr( AdjacencyMatrixType::isSymmetric() )
               graph.getAdjacencyMatrix().reduceAllRows( symmetric_fetch, Plus{}, keep, (IndexType) 0, launchConfig );
            else
               graph.getAdjacencyMatrix().reduceAllRows( fetch, Plus{}, keep, (IndexType) 0, launchConfig );

            // NOTE: These sequential loops over all vertices cause host-device synchronization
            // on GPU backends and scale poorly for large graphs. Consider replacing with
            // parallel reduce in a future optimization.
            // If any active vertex was reached more than once, the graph
            // contains a cycle or cross-edge and is not a tree/forest.
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
         bool foundNext = false;
         for( IndexType i = 0; i < n; i++ ) {
            if( isActive( i ) && visited[ i ] == 0 ) {
               start = i;
               foundNext = true;
               break;
            }
         }
         if( ! foundNext )
            return true;
      }
   }
}

// isTree overloads

template< typename Graph >
bool
isTree( const Graph& graph, typename Graph::IndexType start, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      },
      launchConfig );
}

template< typename Graph, typename EdgePredicate, typename >
bool
isTree(
   const Graph& graph,
   typename Graph::IndexType start,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename >
bool
isTree(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename EdgePredicate, typename >
bool
isTree(
   const Graph& graph,
   typename Graph::IndexType start,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
   return isTree_impl( graph, roots, TreeType::Tree, isActive, std::forward< EdgePredicate >( edgePredicate ), launchConfig );
}

template< typename Graph, typename VertexPredicate >
bool
isTreeIf(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      },
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename EdgePredicate >
bool
isTreeIf(
   const Graph& graph,
   typename Graph::IndexType start,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isTree edge predicate must return bool and accept (source, target, weight)." );

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   Containers::Vector< IndexType > roots( 1, start );
   return isTree_impl( graph, roots, TreeType::Tree, predicate, std::forward< EdgePredicate >( edgePredicate ), launchConfig );
}

// isForest overloads (auto roots)

template< typename Graph >
bool
isForest( const Graph& graph, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      },
      launchConfig );
}

template< typename Graph, typename EdgePredicate, typename >
bool
isForest( const Graph& graph, EdgePredicate&& edgePredicate, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename >
bool
isForest( const Graph& graph, const VertexIndexes& vertexIndexes, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename EdgePredicate, typename >
bool
isForest(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
   return isTree_impl( graph, roots, TreeType::Forest, isActive, std::forward< EdgePredicate >( edgePredicate ), launchConfig );
}

template< typename Graph, typename VertexPredicate >
bool
isForestIf( const Graph& graph, VertexPredicate&& vertexPredicate, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      },
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename EdgePredicate >
bool
isForestIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;

   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isForest edge predicate must return bool and accept (source, target, weight)." );

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   Containers::Vector< IndexType > roots;
   return isTree_impl(
      graph, roots, TreeType::Forest, predicate, std::forward< EdgePredicate >( edgePredicate ), launchConfig );
}

// isForestWithRoots overloads (explicit roots)

template< typename Graph, typename Vector >
bool
isForestWithRoots( const Graph& graph, const Vector& roots, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      },
      launchConfig );
}

template< typename Graph, typename EdgePredicate, typename Vector, typename >
bool
isForestWithRoots(
   const Graph& graph,
   EdgePredicate&& edgePredicate,
   const Vector& roots,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      std::forward< EdgePredicate >( edgePredicate ),
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename Vector, typename >
bool
isForestWithRoots(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   const Vector& roots,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      },
      launchConfig );
}

template< typename Graph, typename VertexIndexes, typename EdgePredicate, typename Vector, typename >
bool
isForestWithRoots(
   const Graph& graph,
   const VertexIndexes& vertexIndexes,
   EdgePredicate&& edgePredicate,
   const Vector& roots,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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

   return isTree_impl( graph, roots, TreeType::Forest, isActive, std::forward< EdgePredicate >( edgePredicate ), launchConfig );
}

template< typename Graph, typename VertexPredicate, typename Vector >
bool
isForestWithRootsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   const Vector& roots,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
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
      },
      launchConfig );
}

template< typename Graph, typename VertexPredicate, typename EdgePredicate, typename Vector >
bool
isForestWithRootsIf(
   const Graph& graph,
   VertexPredicate&& vertexPredicate,
   EdgePredicate&& edgePredicate,
   const Vector& roots,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   static_assert(
      detail::isTreeEdgePredicate_v< EdgePredicate, Graph >,
      "isForestWithRootsIf edge predicate must return bool and accept (source, target, weight)." );

   auto predicate = std::forward< VertexPredicate >( vertexPredicate );
   return isTree_impl(
      graph, roots, TreeType::Forest, predicate, std::forward< EdgePredicate >( edgePredicate ), launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
