// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/find.h>
#include <TNL/Matrices/MatrixBase.h>

namespace TNL::Graphs {

enum class TreeType
{
   Tree,
   Forest
};

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

template< typename Graph, typename Vector >
bool
isTree_impl( const Graph& graph, const Vector& roots, TreeType treeType = TreeType::Tree )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using IndexVectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
   using MatrixType = typename Graph::MatrixType;

   const IndexType n = graph.getNodeCount();

   /////
   // Check if the number of edges is n - 1, i.e number of vertexes - 1 if we test for tree.
   if( treeType == TreeType::Tree && graph.getEdgeCount() != n - 1 )
      return false;

   IndexVectorType visited( n, 0 ), visited_old( n, -1 ), parents( n, 0 );
   IndexType start_node = 0, rootsIdx = 0;
   if( ! roots.empty() )
      start_node = roots.getElement( rootsIdx++ );
   while( true ) {
      visited.setElement( start_node, 1 );
      if( std::is_same_v< DeviceType, Devices::Sequential > ) {
         std::queue< IndexType > q;
         q.push( start_node );
         while( ! q.empty() ) {
            IndexType current = q.front();
            q.pop();
            const auto row = graph.getAdjacencyMatrix().getRow( current );
            for( IndexType i = 0; i < row.getSize(); i++ ) {
               const auto& neighbor = row.getColumnIndex( i );
               if( neighbor == Matrices::paddingIndex< IndexType > )
                  continue;
               if( ! visitNeighbour( current, neighbor, visited, parents, q ) )
                  return false;
            }
            if constexpr( MatrixType::isSymmetric() ) {  // search the adjacency matrix for other neighbours
               for( IndexType rowIdx = 0; rowIdx < graph.getNodeCount(); rowIdx++ ) {
                  if( rowIdx == current )
                     continue;
                  auto row = graph.getAdjacencyMatrix().getRow( rowIdx );
                  for( IndexType i = 0; i < row.getSize(); i++ ) {
                     const auto& col = row.getColumnIndex( i );
                     if( col == Matrices::paddingIndex< IndexType > || col != current )
                        continue;
                     if( ! visitNeighbour( current, rowIdx, visited, parents, q ) )
                        return false;
                  }
               }
            }
         }
      }
      else {
         while( visited_old != visited ) {
            visited_old = visited;
            auto visited_view = visited.getView();
            auto visited_old_view = visited_old.getView();
            // NVCC does not support constepxr if inside a lambda
            auto symmetric_fetch =
               [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
            {
               if( ! visited_old_view[ columnIdx ] )
                  Algorithms::AtomicOperations< DeviceType >::add( visited_view[ columnIdx ], visited_old_view[ rowIdx ] );
               if( visited_old_view[ rowIdx ] )
                  return 0;
               return visited_old_view[ columnIdx ] != 0;
            };
            auto fetch =
               [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) mutable -> IndexType
            {
               if( visited_old_view[ rowIdx ] )
                  return 0;
               return visited_old_view[ columnIdx ] != 0;
            };
            auto keep = [ = ] __cuda_callable__( IndexType rowIdx, const IndexType value ) mutable
            {
               visited_view[ rowIdx ] = visited_view[ rowIdx ] + value;
            };
            if constexpr( MatrixType::isSymmetric() )
               graph.getAdjacencyMatrix().reduceAllRows( symmetric_fetch, TNL::Plus{}, keep, (IndexType) 0 );
            else
               graph.getAdjacencyMatrix().reduceAllRows( fetch, TNL::Plus{}, keep, (IndexType) 0 );

            if( max( visited ) > 1 )
               return false;
            if( min( visited ) == 1 )
               return true;
         }
      }
      if( min( visited ) == 1 )
         return true;
      if( treeType == TreeType::Tree )
         return false;
      if( ! roots.empty() ) {
         if( rootsIdx < roots.getSize() )
            start_node = roots.getElement( rootsIdx++ );
         else
            return false;
      }
      else
         start_node = Algorithms::find( visited, 0 ).second;
   }
}

template< typename Graph >
bool
isTree( const Graph& graph, typename Graph::IndexType start_node = 0 )
{
   using IndexType = typename Graph::IndexType;

   Containers::Vector< IndexType > roots( 1, start_node );
   return isTree_impl( graph, roots, TreeType::Tree );
}

template< typename Graph, typename Vector >
bool
isForest( const Graph& graph, const Vector& roots )
{
   return isTree_impl( graph, roots, TreeType::Forest );
}

template< typename Graph >
bool
isForest( const Graph& graph )
{
   Containers::Vector< typename Graph::IndexType > roots;
   return isTree_impl( graph, roots, TreeType::Forest );
}

}  // namespace TNL::Graphs
