// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <map>
#include <ostream>
#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/reduce.h>

namespace TNL::Graphs {

struct Directed {};

struct Undirected {};

template< typename Matrix,
          typename GraphType_ = Directed >
          // std::enable_if_t< IsMatrixType< Matrix >::value, bool > = true > TODO: fix check for matrix type
struct Graph
{
   using MatrixType = Matrix;
   using IndexType = typename Matrix::IndexType;
   using DeviceType = typename Matrix::DeviceType;
   using ValueType = typename Matrix::RealType;
   using GraphType = GraphType_;

   static constexpr bool isDirected() { return std::is_same_v< GraphType, Directed >; }

   static constexpr bool isUndirected() { return std::is_same_v< GraphType, Undirected >; }

   Graph() = default;

   Graph( const MatrixType& matrix ) {
      this->adjacencyMatrix = matrix;
   }

   Graph( MatrixType&& matrix )
      : MatrixType( std::move( matrix ) ) {}

   Graph( const Graph& ) = default;

   Graph( Graph&& ) = default;

   template< typename OtherGraph >
   Graph( const OtherGraph& other ) {
      this->adjacencyMatrix = other.getAdjacencyMatrix();
   }

   template< typename OtherGraph >
   Graph( const OtherGraph&& other )
      : MatrixType( std::forward< typename OtherGraph::MatrixType >( other.getAdjacencyMatrix() ) ) {}

   Graph( IndexType nodesCount,
          const std::initializer_list< std::tuple< IndexType, IndexType, ValueType > >& data ) {

      if( isUndirected() && ! MatrixType::isSymmetric() )  {
         std::map< std::pair< IndexType, IndexType >, ValueType > symmetric_map;
         for( const auto& [source, target, weight] : data ) {
            symmetric_map[ { source, target } ] = weight;
            symmetric_map[ { target, source } ] = weight;
         }
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( symmetric_map );
      }
      else{
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( data );
      }
   }

   template< typename MapIndex, typename MapValue >
   Graph( IndexType nodesCount,
          const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map ) {
      if( isUndirected() && ! MatrixType::isSymmetric() ){
         std::map< std::pair< MapIndex, MapIndex >, MapValue > symmetric_map;
         for( const auto& [key, value] : map ) {
            symmetric_map[ { key.second, key.first } ] = value;
            symmetric_map[ { key.first, key.second } ] = value;
         }
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( symmetric_map );
      }
      else {
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( map );
      }
   }

   Graph& operator=( const Graph& ) = default;

   Graph& operator=( Graph&& ) = default;

   bool operator==( const Graph& other ) const {
      return adjacencyMatrix == other.adjacencyMatrix;
   }

   void setNodeCount( IndexType nodesCount ) {
      adjacencyMatrix.setDimensions( nodesCount, nodesCount );
   }

   template< typename MapIndex, typename MapValue >
   void setEdges( const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map ) {
      this->adjacencyMatrix.setElements( map );
   }

   IndexType getNodeCount() const {
      return adjacencyMatrix.getRows();
   }

   IndexType getEdgeCount() const {
      if constexpr( isUndirected() )
         return adjacencyMatrix.getNonzeroElementsCount() / 2;
      return adjacencyMatrix.getNonzeroElementsCount();
   }

   template< typename Vector >
   void setNodeCapacities( const Vector& nodeCapacities ) {
      adjacencyMatrix.setRowCapacities( nodeCapacities );
   }

   const MatrixType& getAdjacencyMatrix() const {
      return adjacencyMatrix;
   }

   MatrixType& getAdjacencyMatrix() {
      return adjacencyMatrix;
   }

   template< typename Matrix_ >
   void setAdjacencyMatrix( const Matrix_& matrix ) {
      TNL_ASSERT_EQ( matrix.getRows(), matrix.getColumns(), "Adjacency matrix must be square matrix." );
      adjacencyMatrix = matrix;
   }

   template< typename Matrix_ >
   void setAdjacencyMatrix( Matrix_&& matrix ) {
      TNL_ASSERT_EQ( matrix.getRows(), matrix.getColumns(), "Adjacency matrix must be square matrix." );
      adjacencyMatrix = std::move( matrix );
   }

   ValueType getTotalWeight() const {
      auto values_view = adjacencyMatrix.getValues().getConstView();
      auto column_indexes_view = adjacencyMatrix.getColumnIndexes().getConstView();
      const auto padding = adjacencyMatrix.getPaddingIndex();
      ValueType w = Algorithms::reduce< DeviceType >( 0, values_view.getSize(),
         [=] __cuda_callable__ ( IndexType i ) {
            if( column_indexes_view[ i ] != padding )
               return values_view[ i ];
            return ( ValueType ) 0; },
         TNL::Plus{} );
      if constexpr( isUndirected() )
         return 0.5 * w;
      return w;
   }

   ~Graph() = default;

   protected:

   MatrixType adjacencyMatrix;
};

template< typename Matrix, typename GraphType >
std::ostream& operator<<( std::ostream& os, const Graph< Matrix, GraphType >& graph ) {
   os << graph.getAdjacencyMatrix();
   return os;
};

} // namespace TNL::Graphs
