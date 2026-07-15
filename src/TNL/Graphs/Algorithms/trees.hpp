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
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Matrices/MatrixBase.h>

#include "details/activeVertices.hpp"
#include "details/lambdaTraits.hpp"
#include "trees.h"

namespace TNL::Graphs::Algorithms {

enum class TreeType : std::uint8_t
{
   Tree,
   Forest
};

namespace detail {

template< typename Graph >
typename Graph::IndexType
countActiveEdges( const Graph& graph, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using AdjacencyMatrixType = typename Graph::AdjacencyMatrixType;
   const IndexType n = graph.getVertexCount();
   const auto& matrix = graph.getAdjacencyMatrix();
   const auto graphView = graph.getConstView();
   constexpr bool isUndirected = Graph::isUndirected();
   constexpr bool isSymmetric = AdjacencyMatrixType::isSymmetric();

   if constexpr( std::is_same_v< DeviceType, Devices::Sequential > ) {
      IndexType edgeCount = 0;
      for( IndexType rowIdx = 0; rowIdx < n; rowIdx++ ) {
         if( ! graphView.vertexExists( rowIdx ) )
            continue;
         const auto row = matrix.getRow( rowIdx );
         for( IndexType i = 0; i < row.getSize(); i++ ) {
            const auto col = row.getColumnIndex( i );
            if( col == Matrices::paddingIndex< IndexType > )
               continue;
            if( ! graphView.vertexExists( col ) )
               continue;
            const ValueType weight = row.getValue( i );
            if( ! graphView.edgeExists( rowIdx, col, weight ) )
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
            if( ! graphView.vertexExists( rowIdx ) )
               continue;
            const auto row = matrix.getRow( rowIdx );
            for( IndexType i = 0; i < row.getSize(); i++ ) {
               const auto col = row.getColumnIndex( i );
               if( col == Matrices::paddingIndex< IndexType > || col != rowIdx )
                  continue;
               if( ! graphView.edgeExists( rowIdx, col, row.getValue( i ) ) )
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

      auto fetch_edge =
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
      {
         if( columnIdx == Matrices::paddingIndex< IndexType > )
            return 0;
         if( ! graphView.vertexExists( rowIdx ) || ! graphView.vertexExists( columnIdx ) )
            return 0;
         if( ! graphView.edgeExists( rowIdx, columnIdx, value ) )
            return 0;
         if( isUndirected && ! isSymmetric ) {
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

      IndexType total = TNL::Algorithms::reduce< DeviceType, IndexType, IndexType >(
         0,
         n,
         [ = ] __cuda_callable__( IndexType idx ) -> IndexType
         {
            return edgeCountsView[ idx ];
         },
         Plus{},
         (IndexType) 0 );

      if constexpr( isUndirected ) {
         Containers::Vector< IndexType, DeviceType, IndexType > diagonalCounts( n, 0 );
         auto diagView = diagonalCounts.getView();
         auto diag_fetch =
            [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
         {
            if( columnIdx != rowIdx || columnIdx == Matrices::paddingIndex< IndexType > )
               return 0;
            if( ! graphView.vertexExists( rowIdx ) )
               return 0;
            if( ! graphView.edgeExists( rowIdx, columnIdx, value ) )
               return 0;
            return 1;
         };
         auto diag_keep = [ = ] __cuda_callable__( IndexType rowIdx, const IndexType value ) mutable
         {
            diagView[ rowIdx ] = value;
         };
         matrix.reduceAllRows( diag_fetch, Plus{}, diag_keep, (IndexType) 0, launchConfig );
         total += TNL::Algorithms::reduce< DeviceType, IndexType, IndexType >(
            0,
            n,
            [ = ] __cuda_callable__( IndexType idx ) -> IndexType
            {
               return diagView[ idx ];
            },
            Plus{},
            (IndexType) 0 );
      }
      return total;
   }
}

template< typename Graph >
typename Graph::IndexType
countActiveVertices( const Graph& graph, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   using DeviceType = typename Graph::DeviceType;
   const IndexType n = graph.getVertexCount();
   const auto graphView = graph.getConstView();

   return TNL::Algorithms::reduce< DeviceType, IndexType, IndexType >(
      0,
      n,
      [ = ] __cuda_callable__( IndexType idx ) -> IndexType
      {
         return graphView.vertexExists( idx ) ? 1 : 0;
      },
      Plus{},
      (IndexType) 0 );
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

template< typename Graph, typename Vector >
bool
isTree_impl(
   const Graph& graph,
   const Vector& roots,
   TreeType treeType,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
   using AdjacencyMatrixType = typename Graph::AdjacencyMatrixType;

   const IndexType n = graph.getVertexCount();
   const auto graphView = graph.getConstView();

   if( treeType == TreeType::Tree ) {
      // A tree on n vertices has exactly n-1 edges.  Guard against the
      // unsigned underflow when nActive == 0 (an empty graph is trivially
      // a tree — zero vertices, zero edges).
      const IndexType nActive = detail::countActiveVertices( graph, launchConfig );
      if( nActive == 0 )
         return true;
      const IndexType activeEdgeCount = detail::countActiveEdges( graph, launchConfig );
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
      start = TNL::Algorithms::reduce< DeviceType, IndexType, IndexType >(
         0,
         n,
         [ = ] __cuda_callable__( IndexType i ) -> IndexType
         {
            return graphView.vertexExists( i ) ? i : std::numeric_limits< IndexType >::max();
         },
         TNL::Min{},
         std::numeric_limits< IndexType >::max() );
      if( start == std::numeric_limits< IndexType >::max() )
         return treeType == TreeType::Forest;
   }
   // BFS from the start vertex.  If TreeType::Tree, we return false as
   // soon as a cycle/cross-edge is found or not all active vertices are
   // reached.  For TreeType::Forest we restart from the next unvisited
   // active vertex (or the next explicit root) until all are covered.
   while( true ) {
      bool startActive = TNL::Algorithms::reduce< DeviceType, IndexType, bool >(
         0,
         1,
         [ = ] __cuda_callable__( IndexType ) -> bool
         {
            return graphView.vertexExists( start );
         },
         TNL::LogicalAnd{},
         true );
      if( ! startActive )
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
               if( ! graphView.vertexExists( neighbor ) )
                  continue;
               const ValueType weight = row.getValue( i );
               if( ! graphView.edgeExists( current, neighbor, weight ) )
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
                  if( ! graphView.vertexExists( rowIdx ) )
                     continue;
                  auto row2 = graph.getAdjacencyMatrix().getRow( rowIdx );
                  for( IndexType i = 0; i < row2.getSize(); i++ ) {
                     const auto col = row2.getColumnIndex( i );
                     if( col == Matrices::paddingIndex< IndexType > || col != current )
                        continue;
                     const ValueType weight = row2.getValue( i );
                     if( ! graphView.edgeExists( rowIdx, current, weight ) )
                        continue;
                     if( ! visitNeighbor( current, rowIdx, visited, parents, q ) )
                        return false;
                  }
               }
            }
         }
      }
      else {
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
               if( ! graphView.vertexExists( columnIdx ) || ! graphView.vertexExists( rowIdx ) )
                  return 0;
               if( ! graphView.edgeExists( rowIdx, columnIdx, value ) )
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
               if( ! graphView.vertexExists( columnIdx ) || ! graphView.vertexExists( rowIdx ) )
                  return 0;
               if( ! graphView.edgeExists( rowIdx, columnIdx, value ) )
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
               graph.getAdjacencyMatrix().reduceAllRows(
                  symmetric_fetch, Plus{}, keep, static_cast< IndexType >( 0 ), launchConfig );
            else
               graph.getAdjacencyMatrix().reduceAllRows( fetch, Plus{}, keep, static_cast< IndexType >( 0 ), launchConfig );

            bool anyExceeded = ! TNL::Algorithms::reduce< DeviceType, IndexType, bool >(
               0,
               n,
               [ = ] __cuda_callable__( IndexType i ) -> bool
               {
                  return ! ( graphView.vertexExists( i ) && visitedView[ i ] > 1 );
               },
               TNL::LogicalAnd{},
               true );
            if( anyExceeded )
               return false;
            bool allVisited = TNL::Algorithms::reduce< DeviceType, IndexType, bool >(
               0,
               n,
               [ = ] __cuda_callable__( IndexType i ) -> bool
               {
                  return ! graphView.vertexExists( i ) || visitedView[ i ] == 1;
               },
               TNL::LogicalAnd{},
               true );
            if( allVisited )
               return true;
         }
      }
      auto visitedViewOuter = visited.getConstView();
      bool allVisitedSeq = TNL::Algorithms::reduce< DeviceType, IndexType, bool >(
         0,
         n,
         [ = ] __cuda_callable__( IndexType i ) -> bool
         {
            return ! graphView.vertexExists( i ) || visitedViewOuter[ i ] == 1;
         },
         TNL::LogicalAnd{},
         true );
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
         start = TNL::Algorithms::reduce< DeviceType, IndexType, IndexType >(
            0,
            n,
            [ = ] __cuda_callable__( IndexType i ) -> IndexType
            {
               return ( graphView.vertexExists( i ) && visitedViewOuter[ i ] == 0 ) ? i
                                                                                    : std::numeric_limits< IndexType >::max();
            },
            TNL::Min{},
            std::numeric_limits< IndexType >::max() );
         if( start == std::numeric_limits< IndexType >::max() )
            return true;
      }
   }
}

template< typename Graph >
bool
isTree( const Graph& graph, typename Graph::IndexType start, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   Containers::Vector< IndexType > roots( 1, start );
   return isTree_impl( graph, roots, TreeType::Tree, launchConfig );
}

template< typename Graph >
bool
isForest( const Graph& graph, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   Containers::Vector< IndexType > roots;
   return isTree_impl( graph, roots, TreeType::Forest, launchConfig );
}

template< typename Graph, typename Vector >
bool
isForestWithRoots( const Graph& graph, const Vector& roots, TNL::Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return isTree_impl( graph, roots, TreeType::Forest, launchConfig );
}

}  // namespace TNL::Graphs::Algorithms
