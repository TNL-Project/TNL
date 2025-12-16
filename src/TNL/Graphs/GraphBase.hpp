// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Matrices/SparseMatrix.h>
#include "Graph.h"

namespace TNL::Graphs {

template< typename AdjacencyMatrixView, typename GraphType_ >
constexpr bool
GraphBase< AdjacencyMatrixView, GraphType_ >::isDirected()
{
   return std::is_same_v< GraphType, DirectedGraph >;
}

template< typename AdjacencyMatrixView, typename GraphType_ >
constexpr bool
GraphBase< AdjacencyMatrixView, GraphType_ >::isUndirected()
{
   return std::is_same_v< GraphType_, UndirectedGraph >;
}

template< typename AdjacencyMatrixView, typename GraphType_ >
bool
GraphBase< AdjacencyMatrixView, GraphType_ >::operator==( const GraphBase& other ) const
{
   return adjacencyMatrixView == other.adjacencyMatrixView;
}

template< typename AdjacencyMatrixView, typename GraphType_ >
[[nodiscard]] std::string
GraphBase< AdjacencyMatrixView, GraphType_ >::getSerializationType()
{
   return "Graph< " + AdjacencyMatrixView::getSerializationType() + ", " + GraphType::getSerializationType() + " >";
}

template< typename AdjacencyMatrixView, typename GraphType_ >
[[nodiscard]] __cuda_callable__
auto
GraphBase< AdjacencyMatrixView, GraphType_ >::getAdjacencyMatrix() const -> const AdjacencyMatrixView&
{
   return adjacencyMatrixView;
}

template< typename AdjacencyMatrixView, typename GraphType_ >
[[nodiscard]] __cuda_callable__
auto
GraphBase< AdjacencyMatrixView, GraphType_ >::getNodeCount() const -> IndexType
{
   return adjacencyMatrixView.getRows();
}

template< typename AdjacencyMatrixView, typename GraphType_ >
[[nodiscard]] __cuda_callable__
auto
GraphBase< AdjacencyMatrixView, GraphType_ >::getEdgeCount() const -> IndexType
{
   if constexpr( isUndirected() ) {
      auto diagonalEntries = sum( notEqualTo( Matrices::getDiagonal( adjacencyMatrixView ), 0 ) );
      return ( adjacencyMatrixView.getNonzeroElementsCount() - diagonalEntries ) / 2 + diagonalEntries;
   }
   else {
      return adjacencyMatrixView.getNonzeroElementsCount();
   }
}

template< typename AdjacencyMatrixView, typename GraphType_ >
[[nodiscard]] __cuda_callable__
auto
GraphBase< AdjacencyMatrixView, GraphType_ >::getNode( IndexType nodeIdx ) const -> ConstNodeView
{
   return { this->getAdjacencyMatrix().getSegments().getSegmentView( nodeIdx ),
            this->getAdjacencyMatrix().getValues().getView() };
}

template< typename AdjacencyMatrixView, typename GraphType_ >
[[nodiscard]] __cuda_callable__
auto
GraphBase< AdjacencyMatrixView, GraphType_ >::getNode( IndexType nodeIdx ) -> NodeView
{
   return { this->getAdjacencyMatrix().getSegments().getSegmentView( nodeIdx ),
            this->getAdjacencyMatrix().getValues().getView() };
}

template< typename AdjacencyMatrixView, typename GraphType_ >
__cuda_callable__
void
GraphBase< AdjacencyMatrixView, GraphType_ >::setEdgeWeight( IndexType nodeIdx, IndexType edgeIdx, const ValueType& value )
{
   this->getAdjacencyMatrix().setElement( nodeIdx, edgeIdx, value );
}

template< typename AdjacencyMatrixView, typename GraphType_ >
[[nodiscard]] __cuda_callable__
auto
GraphBase< AdjacencyMatrixView, GraphType_ >::getEdgeWeight( IndexType nodeIdx, IndexType edgeIdx ) const -> ValueType
{
   return this->getAdjacencyMatrix().getElement( nodeIdx, edgeIdx );
}

template< typename AdjacencyMatrixView, typename GraphType_ >
std::ostream&
operator<<( std::ostream& os, const GraphBase< AdjacencyMatrixView, GraphType_ >& graph )
{
   os << graph.getAdjacencyMatrix();
   return os;
}

template< typename AdjacencyMatrixView, typename GraphType_ >
File&
operator<<( File& file, const GraphBase< AdjacencyMatrixView, GraphType_ >& graph )
{
   saveObjectType( file, graph.getSerializationType() );
   return file << graph.getAdjacencyMatrix();
}

template< typename AdjacencyMatrixView, typename GraphType_ >
File&
operator<<( File&& file, const GraphBase< AdjacencyMatrixView, GraphType_ >& graph )
{
   return file << graph;
}

}  // namespace TNL::Graphs
