// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "Graph.h"

namespace TNL::Graphs {

template< typename Matrix, GraphTypes GraphType >
constexpr bool
Graph< Matrix, GraphType >::isDirected()
{
   return getGraphType() == GraphTypes::Directed;
}

template< typename Matrix, GraphTypes GraphType >
constexpr bool
Graph< Matrix, GraphType >::isUndirected()
{
   return getGraphType() == GraphTypes::Undirected;
}

template< typename Matrix, GraphTypes GraphType >
constexpr GraphTypes
Graph< Matrix, GraphType >::getGraphType()
{
   return GraphType;
}

template< typename Matrix, GraphTypes GraphType >
Graph< Matrix, GraphType >::Graph( const MatrixType& matrix )
{
   this->adjacencyMatrix = matrix;
}

template< typename Matrix, GraphTypes GraphType >
Graph< Matrix, GraphType >::Graph( MatrixType&& matrix )
: MatrixType( std::move( matrix ) )
{}

template< typename Matrix, GraphTypes GraphType >
template< typename OtherGraph >
Graph< Matrix, GraphType >::Graph( const OtherGraph& other )
{
   this->adjacencyMatrix = other.getAdjacencyMatrix();
}

template< typename Matrix, GraphTypes GraphType >
template< typename OtherGraph >
Graph< Matrix, GraphType >::Graph( const OtherGraph&& other )
: MatrixType( std::forward< typename OtherGraph::MatrixType >( other.getAdjacencyMatrix() ) )
{}

template< typename Matrix, GraphTypes GraphType >
Graph< Matrix, GraphType >::Graph( IndexType nodesCount,
                                   const std::initializer_list< std::tuple< IndexType, IndexType, ValueType > >& data,
                                   Matrices::SymmetricMatrixEncoding encoding )
{
   if( isUndirected() && ! MatrixType::isSymmetric() ) {
      std::map< std::pair< IndexType, IndexType >, ValueType > symmetric_map;
      for( const auto& [ source, target, weight ] : data ) {
         symmetric_map[ { source, target } ] = weight;
         symmetric_map[ { target, source } ] = weight;
      }
      this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
      this->adjacencyMatrix.setElements( symmetric_map );
   }
   else {
      this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
      this->adjacencyMatrix.setElements( data, encoding );
   }
}

template< typename Matrix, GraphTypes GraphType >
template< typename MapIndex, typename MapValue >
Graph< Matrix, GraphType >::Graph( IndexType nodesCount,
                                   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
                                   Matrices::SymmetricMatrixEncoding encoding )
{
   if( isUndirected() && ! MatrixType::isSymmetric() ) {
      std::map< std::pair< MapIndex, MapIndex >, MapValue > symmetric_map;
      for( const auto& [ key, value ] : map ) {
         symmetric_map[ { key.second, key.first } ] = value;
         symmetric_map[ { key.first, key.second } ] = value;
      }
      this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
      this->adjacencyMatrix.setElements( symmetric_map );
   }
   else {
      this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
      this->adjacencyMatrix.setElements( map, encoding );
   }
}

template< typename Matrix, GraphTypes GraphType >
bool
Graph< Matrix, GraphType >::operator==( const Graph& other ) const
{
   return adjacencyMatrix == other.adjacencyMatrix;
}

template< typename Matrix, GraphTypes GraphType >
void
Graph< Matrix, GraphType >::setNodeCount( IndexType nodesCount )
{
   adjacencyMatrix.setDimensions( nodesCount, nodesCount );
}

template< typename Matrix, GraphTypes GraphType >
template< typename MapIndex, typename MapValue >
void
Graph< Matrix, GraphType >::setEdges( const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map )
{
   this->adjacencyMatrix.setElements( map );
}

template< typename Matrix, GraphTypes GraphType >
[[nodiscard]] auto
Graph< Matrix, GraphType >::getNodeCount() const -> IndexType
{
   return adjacencyMatrix.getRows();
}

template< typename Matrix, GraphTypes GraphType >
[[nodiscard]] auto
Graph< Matrix, GraphType >::getEdgeCount() const -> IndexType
{
   if constexpr( isUndirected() )
      return adjacencyMatrix.getNonzeroElementsCount() / 2;
   return adjacencyMatrix.getNonzeroElementsCount();
}

template< typename Matrix, GraphTypes GraphType >
template< typename Vector >
void
Graph< Matrix, GraphType >::setNodeCapacities( const Vector& nodeCapacities )
{
   adjacencyMatrix.setRowCapacities( nodeCapacities );
}

template< typename Matrix, GraphTypes GraphType >
[[nodiscard]] auto
Graph< Matrix, GraphType >::getAdjacencyMatrix() const -> const MatrixType&
{
   return adjacencyMatrix;
}

template< typename Matrix, GraphTypes GraphType >
auto
Graph< Matrix, GraphType >::getAdjacencyMatrix() -> MatrixType&
{
   return adjacencyMatrix;
}

template< typename Matrix, GraphTypes GraphType >
template< typename Matrix_ >
void
Graph< Matrix, GraphType >::setAdjacencyMatrix( Matrix_ matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = std::move( matrix );
}

template< typename Matrix, GraphTypes GraphType >
std::ostream&
operator<<( std::ostream& os, const Graph< Matrix, GraphType >& graph )
{
   os << graph.getAdjacencyMatrix();
   return os;
}

}  // namespace TNL::Graphs
