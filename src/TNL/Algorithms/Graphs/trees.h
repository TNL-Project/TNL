// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <queue>
#include <TNL/Containers/Vector.h>


namespace TNL::Algorithms::Graphs {

enum class TreeType { Tree, Forest };

template< typename Graph >
bool isTree_impl( const Graph& graph, TreeType treeType = TreeType::Tree )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using VectorType = Containers::Vector< ValueType, DeviceType, IndexType >;

   const IndexType n = graph.getNodesCount();
   /////
   // Check if the number of edges is n - 1, i.e number of vertexes - 1 if we test for tree.
   if( treeType == TreeType::Tree ) {
      const IndexType edges_count = graph.getEdgesCount();
      if( edges_count != ( 2*n - 2 ) )
            return false;
   }

   VectorType visited( n , 0 ), aux( n ), parents( n, 0 );
   IndexType start_node = 0;
   while( true ) {
      visited.setElement( start_node, 1 );
      if( std::is_same_v< DeviceType, Devices::Sequential > ) {
         std::queue< IndexType > q;
         q.push( start_node );
         while( !q.empty() ) {
            IndexType current = q.front();
            q.pop();
            const auto row = graph.getAdjacencyMatrix().getRow( current );
            for( IndexType i = 0; i < row.getSize(); i++ ) {
               const auto& neighbor = row.getColumnIndex( i );
               if( neighbor == parents[ current ] )
                  continue;
               if( neighbor == graph.getAdjacencyMatrix().getPaddingIndex() )
                  continue;
               if( visited[ neighbor ] )
                  return false;
               parents[ neighbor ] = current;
               visited[ neighbor ] = 1;
               q.push( neighbor );
            }
         }
      }
      else {
         while( aux != visited )
         {
            aux = visited;
            auto visited_view = visited.getView();
            auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, const ValueType& value ) -> IndexType {
               if( visited_view[ rowIdx ] )
                  return 0;
               return visited_view[ columnIdx ] != 0;
            };
            auto keep = [=] __cuda_callable__ ( IndexType rowIdx, const IndexType value ) mutable {
               visited_view[ rowIdx ] += value;
            };
            graph.getAdjacencyMatrix().reduceAllRows( fetch, TNL::Plus{}, keep, ( IndexType ) 0 );

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
      start_node = visited.find( 0 );
   }
}

template< typename Graph >
bool isTree( const Graph& graph )
{
   return isTree_impl( graph, TreeType::Tree );
}

template< typename Graph >
bool isForest( const Graph& graph )
{
   return isTree_impl( graph, TreeType::Forest );
}

} // namespace TNL::Algorithms::Graphs
