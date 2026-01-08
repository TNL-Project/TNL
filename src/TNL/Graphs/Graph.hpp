// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/SparseMatrix.h>
#include "Graph.h"

namespace TNL::Graphs {

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::Graph( IndexType nodesCount )
: adjacencyMatrix( nodesCount, nodesCount )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::Graph( const AdjacencyMatrixType& matrix )
{
   this->adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
void
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::reset()
{
   this->adjacencyMatrix.reset();
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::Graph( AdjacencyMatrixType&& matrix )
: AdjacencyMatrixType( std::move( matrix ) )
{
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
template< typename OtherGraph >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::Graph( const OtherGraph& other )
{
   // Check if OtherGraph is a Graph type or a Matrix
   if constexpr( decltype( isGraph( std::declval< OtherGraph >() ) )::value ) {
      this->adjacencyMatrix = other.getAdjacencyMatrix();
   }
   else {
      // OtherGraph is assumed to be a matrix type
      this->adjacencyMatrix = other;
   }
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
template< typename OtherGraph >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::Graph( const OtherGraph&& other )
{
   // Check if OtherGraph is a Graph type or a Matrix
   if constexpr( decltype( isGraph( std::declval< OtherGraph >() ) )::value ) {
      adjacencyMatrix = std::forward< typename OtherGraph::AdjacencyMatrixType >( other.getAdjacencyMatrix() );
   }
   else {
      // OtherGraph is assumed to be a matrix type
      adjacencyMatrix = std::forward< OtherGraph >( other );
   }
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::Graph(
   IndexType vertexCount,
   const std::initializer_list< std::tuple< IndexType, IndexType, ValueType > >& data,
   Matrices::MatrixElementsEncoding encoding )
{
   setVertexCount( vertexCount );
   setEdges( data, encoding );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
template< typename MapIndex, typename MapValue >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::Graph(
   IndexType vertexCount,
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
   Matrices::MatrixElementsEncoding encoding )
{
   setVertexCount( vertexCount );
   setEdges( map, encoding );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >&
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::operator=( const Graph& other )
{
   this->adjacencyMatrix = other.adjacencyMatrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
template< typename OtherGraph, std::enable_if_t< isGraph< OtherGraph >( std::declval< OtherGraph >() ) > >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >&
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::operator=( const OtherGraph& other )
{
   this->adjacencyMatrix = other.getAdjacencyMatrix();
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >&
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::operator=( Graph&& other )
{
   this->adjacencyMatrix = std::move( other.adjacencyMatrix );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
auto
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::getView() -> ViewType
{
   return ViewType( this->adjacencyMatrixView );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
auto
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::getConstView() const -> ConstViewType
{
   return ConstViewType( this->adjacencyMatrix.getConstView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
void
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::setVertexCount( IndexType nodesCount )
{
   adjacencyMatrix.setDimensions( nodesCount, nodesCount );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
template< typename Vector >
void
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::setEdgeCounts( const Vector& edgeCounts )
{
   this->adjacencyMatrix.setRowCapacities( edgeCounts );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
void
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::setEdges(
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
            this->getVertexCount(), this->getVertexCount(), symmetric_map );
         this->adjacencyMatrix = tempMatrix;
      }
      else {
         this->adjacencyMatrix.setElements( symmetric_map );
      }
   }
   else {
      if constexpr( Matrices::is_dense_matrix< AdjacencyMatrixType >::value ) {
         Matrices::SparseMatrix< ValueType, typename AdjacencyMatrixType::DeviceType, IndexType > tempMatrix(
            this->getVertexCount(), this->getVertexCount(), data, encoding );
         this->adjacencyMatrix = tempMatrix;
      }
      else {
         this->adjacencyMatrix.setElements( data, encoding );
      }
   }
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
template< typename MapIndex, typename MapValue >
void
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::setEdges(
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
   Matrices::MatrixElementsEncoding encoding )
{
   this->adjacencyMatrix.setDimensions( this->getVertexCount(), this->getVertexCount() );
   if( isUndirected() && ! AdjacencyMatrixType::isSymmetric() ) {
      std::map< std::pair< MapIndex, MapIndex >, MapValue > symmetric_map;
      for( const auto& [ key, value ] : map ) {
         symmetric_map[ { key.second, key.first } ] = value;
         symmetric_map[ { key.first, key.second } ] = value;
      }
      this->adjacencyMatrix.setElements( symmetric_map );
   }
   else
      this->adjacencyMatrix.setElements( map, encoding );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
template< typename Vector >
void
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::setVertexCapacities( const Vector& nodeCapacities )
{
   adjacencyMatrix.setRowCapacities( nodeCapacities );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
[[nodiscard]] auto
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::getAdjacencyMatrix() const -> const AdjacencyMatrixType&
{
   return adjacencyMatrix;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
[[nodiscard]] auto
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::getAdjacencyMatrix() -> AdjacencyMatrixType&
{
   return adjacencyMatrix;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
void
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::setAdjacencyMatrix( const AdjacencyMatrixType& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
void
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::setAdjacencyMatrix( AdjacencyMatrixType&& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = std::forward< AdjacencyMatrixType >( matrix );
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
template< typename Matrix_ >
void
Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >::setAdjacencyMatrix( const Matrix_& matrix )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::logic_error( "Graph: adjacency matrix must be square matrix." );
   adjacencyMatrix = matrix;
   Base::adjacencyMatrixView.bind( this->adjacencyMatrix.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
File&
operator>>( File& file, Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >& graph )
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

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
File&
operator>>( File&& file, Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >& graph )
{
   return file >> graph;
}

}  // namespace TNL::Graphs
