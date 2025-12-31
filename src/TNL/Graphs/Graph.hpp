// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/SparseMatrix.h>
#include "Graph.h"

namespace TNL::Graphs {

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::Graph( IndexType nodesCount )
: adjacencyMatrix( nodesCount, nodesCount )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::Graph( const AdjacencyMatrixType& matrix )
{
   this->adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::Graph( AdjacencyMatrixType&& matrix )
: AdjacencyMatrixType( std::move( matrix ) )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
template< typename OtherGraph >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::Graph( const OtherGraph& other )
{
   // Check if OtherGraph is a Graph type or a Matrix
   if constexpr( decltype(isGraph(std::declval<OtherGraph>()))::value ) {
      this->adjacencyMatrix = other.getAdjacencyMatrix();
   } else {
      // OtherGraph is assumed to be a matrix type
      this->adjacencyMatrix = other;
   }
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
template< typename OtherGraph >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::Graph( const OtherGraph&& other )
{
   // Check if OtherGraph is a Graph type or a Matrix
   if constexpr( decltype(isGraph(std::declval<OtherGraph>()))::value ) {
      adjacencyMatrix = std::forward< typename OtherGraph::AdjacencyMatrixType >( other.getAdjacencyMatrix() );
   } else {
      // OtherGraph is assumed to be a matrix type
      adjacencyMatrix = std::forward< OtherGraph >( other );
   }
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::Graph(
   IndexType nodesCount,
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

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
template< typename MapIndex, typename MapValue >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::Graph(
   IndexType nodesCount,
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

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >&
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::operator=( const Graph& other )
{
   this->adjacencyMatrix = other.adjacencyMatrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
template< typename OtherGraph, std::enable_if_t< isGraph< OtherGraph >( std::declval< OtherGraph >() ) > >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >&
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::operator=( const OtherGraph& other )
{
   this->adjacencyMatrix = other.getAdjacencyMatrix();
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >&
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::operator=( Graph&& other )
{
   this->adjacencyMatrix = std::move( other.adjacencyMatrix );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
auto
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::getView() -> ViewType
{
   return ViewType( this->adjacencyMatrixView );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
auto
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::getConstView() const -> ConstViewType
{
   return ConstViewType( this->adjacencyMatrix.getConstView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
void
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::setVertexCount( IndexType nodesCount )
{
   adjacencyMatrix.setDimensions( nodesCount, nodesCount );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
template< typename MapIndex, typename MapValue >
void
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::setEdges(
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map )
{
   if constexpr( isUndirected() ) {
      this->adjacencyMatrix.setElements( map, Matrices::MatrixElementsEncoding::SymmetricMixed );
   }
   else
      this->adjacencyMatrix.setElements( map );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
template< typename Vector >
void
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::setVertexCapacities( const Vector& nodeCapacities )
{
   adjacencyMatrix.setRowCapacities( nodeCapacities );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] auto
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::getAdjacencyMatrix() const -> const AdjacencyMatrixType&
{
   return adjacencyMatrix;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] auto
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::getAdjacencyMatrix() -> AdjacencyMatrixType&
{
   return adjacencyMatrix;
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
void
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::setAdjacencyMatrix( const AdjacencyMatrixType& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
void
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::setAdjacencyMatrix( AdjacencyMatrixType&& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = std::forward< AdjacencyMatrixType >( matrix );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
template< typename Matrix_ >
void
Graph< Value, Device, Index, Orientation, AdjacencyMatrix >::setAdjacencyMatrix( const Matrix_& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
File&
operator>>( File& file, Graph< Value, Device, Index, Orientation, AdjacencyMatrix >& graph )
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

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
File&
operator>>( File&& file, Graph< Value, Device, Index, Orientation, AdjacencyMatrix >& graph )
{
   return file >> graph;
}

}  // namespace TNL::Graphs
