// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <ostream>
#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Graphs/GraphNodeView.h>
#include <TNL/Graphs/TypeTraits.h>
#include <TNL/Graphs/GraphType.h>
#include <TNL/Matrices/TypeTraits.h>

namespace TNL::Graphs {

/**
 * \brief \e Graph class represents a mathematical graph using an adjacency matrix.
 *
 * \tparam Matrix is type of matrix used to store the adjacency matrix of the graph.
 * \tparam GraphType is type of the graph - directed or undirected.
 */
template< typename Matrix, typename GraphType_ = Graphs::DirectedGraph >
struct GraphBase
{
   static_assert( Matrices::is_matrix_v< Matrix > );

   //! \brief Type of the adjacency matrix.
   using MatrixType = Matrix;

   //! \brief Type of view of the adjacency matrix.
   using MatrixView = typename Matrix::ViewType;

   //! \brief Type of constant view of the adjacency matrix.
   using ConstMatrixView = typename Matrix::ConstViewType;

   //! \brief Type for indexing of the graph nodes.
   using IndexType = typename Matrix::IndexType;

   //! \brief Type of device where the graph will be operating.
   using DeviceType = typename Matrix::DeviceType;

   //! \brief Type for weights of the graph edges.
   using ValueType = typename Matrix::RealType;

   //! \brief Type of the graph - directed or undirected.
   using GraphType = GraphType_;

   using NodeView = GraphNodeView< Graph< MatrixType, GraphType_ > >;

   using ConstNodeView = typename NodeView::ConstNodeView;

   template< typename Matrix_ = MatrixType, typename GraphType__ = GraphType_ >
   using Self = Graph< Matrix, GraphType_ >;

   //! \brief Checks if the graph is directed.
   static constexpr bool
   isDirected();

   //! \brief Checks if the graph is undirected.
   static constexpr bool
   isUndirected();

   //! \brief Default constructor.
   GraphBase() = default;

   //! \brief Copy constructor.
   GraphBase( const GraphBase& ) = default;

   //! \brief Move constructor.
   GraphBase( GraphBase&& ) = default;

   /**
    * \brief Copy-assignment operator.
    *
    * Copy-assignment operator is deleted since it requires data allocation.
    */
   GraphBase&
   operator=( const GraphBase& ) = delete;

   /**
    * \brief Move-assignment operator.
    *
    * Move-assignment operator is deleted since it requires data allocation.
    */
   GraphBase&
   operator=( GraphBase&& ) = delete;

   //! \brief Comparisons operator.
   bool
   operator==( const GraphBase& other ) const;

   //! \brief Returns the type of serialization used for the graph.
   [[nodiscard]] static std::string
   getSerializationType();

   //! \brief Returns the conatnt view of adjacency matrix of the graph.
   [[nodiscard]] __cuda_callable__
   const MatrixView&
   getAdjacencyMatrix() const;

   //! \brief Returns the number of nodes in the graph.
   [[nodiscard]] __cuda_callable__
   IndexType
   getNodeCount() const;

   //! \brief Returns the number of edges in the graph.
   [[nodiscard]] __cuda_callable__
   IndexType
   getEdgeCount() const;

   /***
    * \brief Returns the constant view of the graph node with given index.
    *
    * \param nodeIdx is index of the node to be returned.
    */
   [[nodiscard]] __cuda_callable__
   ConstNodeView
   getNode( IndexType nodeIdx ) const;

   /***
    * \brief Returns the modifiable view of the graph node with given index.
    *
    * \param nodeIdx is index of the node to be returned.
    */
   [[nodiscard]] __cuda_callable__
   NodeView
   getNode( IndexType nodeIdx );

   /***
    * \brief Sets the weight of the edge between given node and its edge index.
    *
    * \param nodeIdx is index of the node.
    * \param edgeIdx is index of the edge of the node.
    * \param value is new weight of the edge.
    */
   __cuda_callable__
   void
   setEdgeWeight( IndexType nodeIdx, IndexType edgeIdx, const ValueType& value );

   /***
    * \brief Returns the weight of the edge between given node and its edge index.
    *
    * \param nodeIdx is index of the node.
    * \param edgeIdx is index of the edge of the node.
    */
   [[nodiscard]] __cuda_callable__
   ValueType
   getEdgeWeight( IndexType nodeIdx, IndexType edgeIdx ) const;

   //! \brief Destructor.
   ~GraphBase() = default;

protected:
   MatrixView adjacencyMatrixView;
};

//! \brief Output stream operator for the \e Graph class.
template< typename Matrix, typename GraphType_ >
std::ostream&
operator<<( std::ostream& os, const GraphBase< Matrix, GraphType_ >& graph );

//! \brief Serialization of graphs into binary files.
template< typename Matrix, typename GraphType_ >
File&
operator<<( File& file, const GraphBase< Matrix, GraphType_ >& graph );

//! \brief Serialization of graphs into binary files.
template< typename Matrix, typename GraphType_ >
File&
operator<<( File&& file, const GraphBase< Matrix, GraphType_ >& graph );

// Note: Deserialization is different for Graph and GraphView,
// see the respective files for implementation.

}  // namespace TNL::Graphs

#include "GraphBase.hpp"
