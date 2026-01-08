// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/SparseMatrix.h>
#include "Graph.h"

namespace TNL::Graphs {

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
constexpr bool
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::isDirected()
{
   return std::is_same_v< Orientation, DirectedGraph >;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
constexpr bool
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::isUndirected()
{
   return std::is_same_v< Orientation, UndirectedGraph >;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
bool
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::operator==( const GraphBase& other ) const
{
   return adjacencyMatrixView == other.adjacencyMatrixView;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] std::string
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::getSerializationType()
{
   return "Graph< " + AdjacencyMatrixView::getSerializationType() + ", " + GraphOrientation::getSerializationType() + " >";
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] __cuda_callable__
auto
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::getAdjacencyMatrixView() -> AdjacencyMatrixView&
{
   return adjacencyMatrixView;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] __cuda_callable__
auto
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::getAdjacencyMatrixView() const -> const AdjacencyMatrixView&
{
   return adjacencyMatrixView;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] __cuda_callable__
auto
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::getVertexCount() const -> IndexType
{
   return adjacencyMatrixView.getRows();
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] __cuda_callable__
auto
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::getEdgeCount() const -> IndexType
{
   if constexpr( isUndirected() ) {
      auto diagonalEntries = sum( notEqualTo( Matrices::getDiagonal( adjacencyMatrixView ), 0 ) );
      return ( adjacencyMatrixView.getNonzeroElementsCount() - diagonalEntries ) / 2 + diagonalEntries;
   }
   else {
      return adjacencyMatrixView.getNonzeroElementsCount();
   }
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] __cuda_callable__
auto
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::getVertex( IndexType vertexIdx ) const -> ConstVertexView
{
   return { adjacencyMatrixView.getRow( vertexIdx ) };
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] __cuda_callable__
auto
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::getVertex( IndexType vertexIdx ) -> VertexView
{
   return { adjacencyMatrixView.getRow( vertexIdx ) };
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
__cuda_callable__
void
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::setEdgeWeight( IndexType vertexIdx,
                                                                                IndexType edgeIdx,
                                                                                const ValueType& value )
{
   adjacencyMatrixView.setElement( vertexIdx, edgeIdx, value );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] __cuda_callable__
auto
GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >::getEdgeWeight( IndexType vertexIdx, IndexType edgeIdx ) const
   -> ValueType
{
   return adjacencyMatrixView.getElement( vertexIdx, edgeIdx );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
std::ostream&
operator<<( std::ostream& os, const GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >& graph )
{
   os << graph.getAdjacencyMatrixView();
   return os;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
File&
operator<<( File& file, const GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >& graph )
{
   saveObjectType( file, graph.getSerializationType() );
   return file << graph.getAdjacencyMatrixView();
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
File&
operator<<( File&& file, const GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >& graph )
{
   return file << graph;
}

}  // namespace TNL::Graphs
