// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>
#include <algorithm>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Graphs/Graph.h>

namespace TNL {
namespace Algorithms {
namespace Graphs {

template< typename Index, typename Real >
bool compareEdges( const Edge< Index, Real > &a, const Edge< Index, Real > &b) {
    return a < b;
}

template< typename Real, typename Index >
struct Forest
{
   Forest( Index size) {
      parent.resize(size);
      rank.resize(size, 0);
      for (Index i = 0; i < size; ++i) {
         parent[i] = i;
      }
   }

   Index getRoot( Index u ) {
      if( parent[ u ] != u )
         parent[ u ] = getRoot( parent[u] );
      return parent[ u ];
   }

   void mergeTrees( Index u, Index v) {
      Index root_u = getRoot( u );
      Index root_v = getRoot( v );

      if( root_u == root_v ) {
            return;
      }

      if( rank[ root_u ] > rank[ root_v ] ) {
         parent[ root_v ] = root_u;
      } else {
         parent[ root_u ] = root_v;
         if( rank[ root_u ] == rank[ root_v ] ) {
            rank[ root_v ]++;
         }
      }
   }

private:
    std::vector< Index > parent;
    std::vector< Index > rank;
};

template< typename Matrix,
          typename Real = typename Matrix::RealType,
          typename Index = typename Matrix::IndexType >
void kruskal(const Matrix& graph, Containers::Array< Edge< Real, Index >, TNL::Devices::Sequential, Index >& minimumSpanningTree )
{
   Index n = graph.getRows();
   std::vector< Edge< Real, Index > > edges;
   for( Index i = 0; i < n; i++ )
   {
      const auto row = graph.getRow(i);
      for( Index j = 0; j < row.getSize(); j++ ) {
         const Index& col = row.getColumnIndex( j );
         if( i < col && col != graph.getPaddingIndex() )
            edges.emplace_back(i, col, row.getValue(j) );
      }
   }

   std::sort(edges.begin(), edges.end(), compareEdges< Real, Index >);

   Forest< Real, Index > forest(n);
   std::vector< Edge< Real, Index > > minSpanningTree;

   for( const auto& edge : edges ) {
      auto u = edge.getSource();
      auto v = edge.getTarget();
      if( forest.getRoot( u ) != forest.getRoot( v ) ) {
         minSpanningTree.push_back( edge );
         forest.mergeTrees( u, v );
      }
   }
   minimumSpanningTree = minSpanningTree;
}

template< typename Matrix,
          typename Real = typename Matrix::RealType,
          typename Index = typename Matrix::IndexType >
void parallelMST(const Matrix& graph )
{
   using Device = typename Matrix::DeviceType;
   using IndexVector = Containers::Vector< Index, Device, Index >;
   using RealVector = Containers::Vector< Real, Device, Index >;
   using RowView = typename Matrix::RowView;

   Index n = graph.getRows();
   IndexVector p( n ), p_old( n, 0 ), q( n );
   RealVector best_edge_weight( n, std::numeric_limits< Real >::max() );
   IndexVector best_edge_target_parent( n, 0 );
   p.forAllElements( [] __cuda_callable__ ( Index& value, Index i ) { value = i; }

   while( p != p_old )
   {
      p_old.swap( p );
      auto p_view = p.getView();
      auto p_old_view = p_old.getView();
      auto best_edge_weight_view = best_edge_weight.getView();
      auto best_edge_target_parent_view = best_edge_target_parent.getView();
      auto hooking = [=] __cuda_callable__ ( RowView& row ) mutable {
         const Index& rowIdx = row.getRowIndex();
         Index minEdgeParent = 0;
         Real minEdgeValue = std::numeric_limits< Real >::max();
         for( Index j = 0; j < row.getSize(); j++ ) {
            const Index& colIdx = row.getColumnIndex( j );
            if( colIdx != graph.getPaddingIndex() && p_old[ colIdx ] != p_old[ rowIdx ] && row.getValue( j ) < minEdgeValue ) {
               minEdgeValue = row.getValue( j );
               minEdgeParent = p_old[ colIdx ];
            }
         }
         best_edge_weight_view[ rowIdx ] = minEdgeValue;
         best_edge_target_parent_view[ rowIdx ] = minEdgeParent;
      };
      graph.forAllRows( hooking );

   }
}

template< typename Matrix,
          typename Real = typename Matrix::RealType,
          typename Index = typename Matrix::IndexType >
void minimumSpanningTree( const Matrix& adjacencyMatrix, Containers::Array< Edge< Real, Index >, typename Matrix::DeviceType >& spanning_tree )
{
   TNL_ASSERT_TRUE( adjacencyMatrix.getRows() == adjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );

   using Device = typename Matrix::DeviceType;

   if constexpr( std::is_same< Device, TNL::Devices::Sequential >::value )
      kruskal( adjacencyMatrix, spanning_tree );
   else
      parallelMST( adjacencyMatrix );
}
} // namespace Graphs
} // namespace Algorithms
} // namespace TNL
