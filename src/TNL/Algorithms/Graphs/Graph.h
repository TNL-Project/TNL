// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <map>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms::Graphs {

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

   Graph( const MatrixType& matrix )
      : MatrixType( matrix ) {}

   Graph( MatrixType&& matrix )
      : MatrixType( std::move( matrix ) ) {}

   Graph( const Graph& ) = default;

   Graph( Graph&& ) = default;

   Graph( IndexType nodesCount,
          const std::initializer_list< std::tuple< IndexType, IndexType, ValueType > >& data ) {
      if( isDirected() || MatrixType::isSymmetric() ) {
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( data );
      }
      else {
         std::map< std::pair< IndexType, IndexType >, ValueType > symmetric_map;
         for( const auto& [source, target, weight] : data ) {
            symmetric_map[ { source, target } ] = weight;
            symmetric_map[ { target, source } ] = weight;
         }
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( symmetric_map );
      }
   }

   template< typename MapIndex, typename MapValue >
   Graph( IndexType nodesCount,
          const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map ) {
      if( isDirected() || MatrixType::isSymmetric() ) {
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( map );
      }
      else {
         std::map< std::pair< MapIndex, MapIndex >, MapValue > symmetric_map;
         for( const auto& [key, value] : map ) {
            symmetric_map[ { key.second, key.first } ] = value;
            symmetric_map[ { key.first, key.second } ] = value;
         }
         this->adjacencyMatrix.setDimensions( nodesCount, nodesCount );
         this->adjacencyMatrix.setElements( symmetric_map );
      }
   }

   Graph& operator=( const Graph& ) = default;

   Graph& operator=( Graph&& ) = default;

   ~Graph() = default;

   IndexType getNodesCount() const {
      return adjacencyMatrix.getRows();
   }

   IndexType getEdgesCount() const {
      return adjacencyMatrix.getNonzeroElementsCount();
   }

   const MatrixType& getAdjacencyMatrix() const {
      return adjacencyMatrix;
   }

   MatrixType& getAdjacencyMatrix() {
      return adjacencyMatrix;
   }

   void setAdjacencyMatrix( const MatrixType& matrix ) {
      adjacencyMatrix = matrix;
   }

   void setAdjacencyMatrix( MatrixType&& matrix ) {
      adjacencyMatrix = std::move( matrix );
   }


   protected:

   MatrixType adjacencyMatrix;
};

} // namespace TNL::Algorithms::Graphs
