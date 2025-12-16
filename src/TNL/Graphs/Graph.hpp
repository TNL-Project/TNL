// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/SparseMatrix.h>
#include "Graph.h"

namespace TNL::Graphs {

template< typename AdjacencyMatrix, typename GraphType_ >
Graph< AdjacencyMatrix, GraphType_ >::Graph( IndexType nodesCount )
: adjacencyMatrix( nodesCount, nodesCount )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
Graph< AdjacencyMatrix, GraphType_ >::Graph( const AdjacencyMatrixType& matrix )
{
   this->adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
Graph< AdjacencyMatrix, GraphType_ >::Graph( AdjacencyMatrixType&& matrix )
: AdjacencyMatrixType( std::move( matrix ) )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
template< typename OtherGraph >
Graph< AdjacencyMatrix, GraphType_ >::Graph( const OtherGraph& other )
{
   this->adjacencyMatrix = other.getAdjacencyMatrix();
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
template< typename OtherGraph >
Graph< AdjacencyMatrix, GraphType_ >::Graph( const OtherGraph&& other )
: adjacencyMatrix( std::forward< typename OtherGraph::MatrixType >( other.getAdjacencyMatrix() ) )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
Graph< AdjacencyMatrix, GraphType_ >::Graph( IndexType nodesCount,
                                             const std::initializer_list< std::tuple< IndexType, IndexType, ValueType > >& data,
                                             Matrices::MatrixElementsEncoding encoding )
{
   if( isUndirected() && ! AdjacencyMatrixType::isSymmetric() ) {
      std::map< std::pair< IndexType, IndexType >, ValueType > symmetric_map;
      for( const auto& [ source, target, weight ] : data ) {
         symmetric_map[ { source, target } ] = weight;
         symmetric_map[ { target, source } ] = weight;
      }
      if constexpr( Matrices::is_dense_matrix< AdjacencyMatrixType >::value ) {
         Matrices::SparseMatrix< ValueType, typename AdjacencyMatrixType::DeviceType, IndexType > tempMatrix(
            nodesCount, nodesCount, symmetric_map );
         this->adjacencyMatrix = tempMatrix;
      }
      else {
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( symmetric_map );
      }
   }
   else {
      if constexpr( Matrices::is_dense_matrix< AdjacencyMatrixType >::value ) {
         Matrices::SparseMatrix< ValueType, typename AdjacencyMatrixType::DeviceType, IndexType > tempMatrix(
            nodesCount, nodesCount, data, encoding );
         this->adjacencyMatrix = tempMatrix;
      }
      else {
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( data, encoding );
      }
   }
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
template< typename MapIndex, typename MapValue >
Graph< AdjacencyMatrix, GraphType_ >::Graph( IndexType nodesCount,
                                             const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
                                             Matrices::MatrixElementsEncoding encoding )
{
   if( isUndirected() && ! AdjacencyMatrixType::isSymmetric() ) {
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
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
Graph< AdjacencyMatrix, GraphType_ >&
Graph< AdjacencyMatrix, GraphType_ >::operator=( const Graph& other )
{
   this->adjacencyMatrix = other.adjacencyMatrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename AdjacencyMatrix, typename GraphType_ >
template< typename OtherGraph, std::enable_if_t< isGraph< OtherGraph >( std::declval< OtherGraph >() ) > >
Graph< AdjacencyMatrix, GraphType_ >&
Graph< AdjacencyMatrix, GraphType_ >::operator=( const OtherGraph& other )
{
   this->adjacencyMatrix = other.getAdjacencyMatrix();
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename AdjacencyMatrix, typename GraphType_ >
Graph< AdjacencyMatrix, GraphType_ >&
Graph< AdjacencyMatrix, GraphType_ >::operator=( Graph&& other )
{
   this->adjacencyMatrix = std::move( other.adjacencyMatrix );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename AdjacencyMatrix, typename GraphType_ >
auto
Graph< AdjacencyMatrix, GraphType_ >::getView() -> ViewType
{
   return ViewType( this->adjacencyMatrixView );
}

template< typename AdjacencyMatrix, typename GraphType_ >
auto
Graph< AdjacencyMatrix, GraphType_ >::getConstView() const -> ConstViewType
{
   return ConstViewType( this->adjacencyMatrix.getConstView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
void
Graph< AdjacencyMatrix, GraphType_ >::setNodeCount( IndexType nodesCount )
{
   adjacencyMatrix.setDimensions( nodesCount, nodesCount );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
template< typename MapIndex, typename MapValue >
void
Graph< AdjacencyMatrix, GraphType_ >::setEdges( const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map )
{
   if constexpr( isUndirected() ) {
      this->adjacencyMatrix.setElements( map, Matrices::MatrixElementsEncoding::SymmetricMixed );
   }
   else
      this->adjacencyMatrix.setElements( map );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
template< typename Vector >
void
Graph< AdjacencyMatrix, GraphType_ >::setNodeCapacities( const Vector& nodeCapacities )
{
   adjacencyMatrix.setRowCapacities( nodeCapacities );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
[[nodiscard]] auto
Graph< AdjacencyMatrix, GraphType_ >::getAdjacencyMatrix() const -> const AdjacencyMatrixType&
{
   return adjacencyMatrix;
}

template< typename AdjacencyMatrix, typename GraphType_ >
[[nodiscard]] auto
Graph< AdjacencyMatrix, GraphType_ >::getAdjacencyMatrix() -> AdjacencyMatrixType&
{
   return adjacencyMatrix;
}

template< typename AdjacencyMatrix, typename GraphType_ >
void
Graph< AdjacencyMatrix, GraphType_ >::setAdjacencyMatrix( const AdjacencyMatrixType& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
void
Graph< AdjacencyMatrix, GraphType_ >::setAdjacencyMatrix( AdjacencyMatrixType&& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = std::forward< AdjacencyMatrixType >( matrix );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
template< typename Matrix_ >
void
Graph< AdjacencyMatrix, GraphType_ >::setAdjacencyMatrix( const Matrix_& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename AdjacencyMatrix, typename GraphType_ >
File&
operator>>( File& file, Graph< AdjacencyMatrix, GraphType_ >& graph )
{
   const std::string type = getObjectType( file );
   if( type != graph.getSerializationType() )
      throw Exceptions::FileDeserializationError( file.getFileName(),
                                                  "object type does not match (expected " + graph.getSerializationType()
                                                     + ", found " + type + ")." );
   AdjacencyMatrix adjacencyMatrix;
   file >> adjacencyMatrix;
   graph.setAdjacencyMatrix( std::move( adjacencyMatrix ) );
   return file;
}

template< typename AdjacencyMatrix, typename GraphType_ >
File&
operator>>( File&& file, Graph< AdjacencyMatrix, GraphType_ >& graph )
{
   return file >> graph;
}

}  // namespace TNL::Graphs
