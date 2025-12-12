// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/SparseMatrix.h>
#include "Graph.h"

namespace TNL::Graphs {

template< typename Matrix, typename GraphType_ >
Graph< Matrix, GraphType_ >::Graph( IndexType nodesCount )
: adjacencyMatrix( nodesCount, nodesCount )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
Graph< Matrix, GraphType_ >::Graph( const MatrixType& matrix )
{
   this->adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
Graph< Matrix, GraphType_ >::Graph( MatrixType&& matrix )
: MatrixType( std::move( matrix ) )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
template< typename OtherGraph >
Graph< Matrix, GraphType_ >::Graph( const OtherGraph& other )
{
   this->adjacencyMatrix = other.getAdjacencyMatrix();
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
template< typename OtherGraph >
Graph< Matrix, GraphType_ >::Graph( const OtherGraph&& other )
: adjacencyMatrix( std::forward< typename OtherGraph::MatrixType >( other.getAdjacencyMatrix() ) )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
Graph< Matrix, GraphType_ >::Graph( IndexType nodesCount,
                                    const std::initializer_list< std::tuple< IndexType, IndexType, ValueType > >& data,
                                    Matrices::MatrixElementsEncoding encoding )
{
   if( isUndirected() && ! MatrixType::isSymmetric() ) {
      std::map< std::pair< IndexType, IndexType >, ValueType > symmetric_map;
      for( const auto& [ source, target, weight ] : data ) {
         symmetric_map[ { source, target } ] = weight;
         symmetric_map[ { target, source } ] = weight;
      }
      if constexpr( Matrices::is_dense_matrix< MatrixType >::value ) {
         Matrices::SparseMatrix< ValueType, typename MatrixType::DeviceType, IndexType > tempMatrix(
            nodesCount, nodesCount, symmetric_map );
         this->adjacencyMatrix = tempMatrix;
      }
      else {
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( symmetric_map );
      }
   }
   else {
      if constexpr( Matrices::is_dense_matrix< MatrixType >::value ) {
         Matrices::SparseMatrix< ValueType, typename MatrixType::DeviceType, IndexType > tempMatrix(
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

template< typename Matrix, typename GraphType_ >
template< typename MapIndex, typename MapValue >
Graph< Matrix, GraphType_ >::Graph( IndexType nodesCount,
                                    const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
                                    Matrices::MatrixElementsEncoding encoding )
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
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
Graph< Matrix, GraphType_ >&
Graph< Matrix, GraphType_ >::operator=( const Graph& other )
{
   this->adjacencyMatrix = other.adjacencyMatrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename Matrix, typename GraphType_ >
template< typename OtherGraph, std::enable_if_t< isGraph< OtherGraph >( std::declval< OtherGraph >() ) > >
Graph< Matrix, GraphType_ >&
Graph< Matrix, GraphType_ >::operator=( const OtherGraph& other )
{
   this->adjacencyMatrix = other.getAdjacencyMatrix();
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename Matrix, typename GraphType_ >
Graph< Matrix, GraphType_ >&
Graph< Matrix, GraphType_ >::operator=( Graph&& other )
{
   this->adjacencyMatrix = std::move( other.adjacencyMatrix );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename Matrix, typename GraphType_ >
bool
Graph< Matrix, GraphType_ >::operator==( const Graph& other ) const
{
   return adjacencyMatrix == other.adjacencyMatrix;
}

template< typename Matrix, typename GraphType_ >
void
Graph< Matrix, GraphType_ >::setNodeCount( IndexType nodesCount )
{
   adjacencyMatrix.setDimensions( nodesCount, nodesCount );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
template< typename MapIndex, typename MapValue >
void
Graph< Matrix, GraphType_ >::setEdges( const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map )
{
   if constexpr( isUndirected() ) {
      this->adjacencyMatrix.setElements( map, Matrices::MatrixElementsEncoding::SymmetricMixed );
   }
   else
      this->adjacencyMatrix.setElements( map );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
template< typename Vector >
void
Graph< Matrix, GraphType_ >::setNodeCapacities( const Vector& nodeCapacities )
{
   adjacencyMatrix.setRowCapacities( nodeCapacities );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
[[nodiscard]] auto
Graph< Matrix, GraphType_ >::getAdjacencyMatrix() const -> const MatrixType&
{
   return adjacencyMatrix;
}

template< typename Matrix, typename GraphType_ >
[[nodiscard]] auto
Graph< Matrix, GraphType_ >::getAdjacencyMatrix() -> MatrixType&
{
   return adjacencyMatrix;
}

template< typename Matrix, typename GraphType_ >
void
Graph< Matrix, GraphType_ >::setAdjacencyMatrix( const MatrixType& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
void
Graph< Matrix, GraphType_ >::setAdjacencyMatrix( MatrixType&& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = std::forward< MatrixType >( matrix );
   Base::adjacencyMatrixView, bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
template< typename Matrix_ >
void
Graph< Matrix, GraphType_ >::setAdjacencyMatrix( const Matrix_& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Matrix, typename GraphType_ >
File&
operator>>( File& file, Graph< Matrix, GraphType_ >& graph )
{
   const std::string type = getObjectType( file );
   if( type != graph.getSerializationType() )
      throw Exceptions::FileDeserializationError( file.getFileName(),
                                                  "object type does not match (expected " + graph.getSerializationType()
                                                     + ", found " + type + ")." );
   Matrix adjacencyMatrix;
   file >> adjacencyMatrix;
   graph.setAdjacencyMatrix( std::move( adjacencyMatrix ) );
   return file;
}

template< typename Matrix, typename GraphType_ >
File&
operator>>( File&& file, Graph< Matrix, GraphType_ >& graph )
{
   return file >> graph;
}

}  // namespace TNL::Graphs
