// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <ostream>
#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Graphs/GraphVertexView.h>
#include <TNL/Graphs/TypeTraits.h>
#include <TNL/Graphs/GraphOrientation.h>
#include <TNL/Matrices/TypeTraits.h>

namespace TNL::Graphs {
/**
 * \brief \e Graph class represents a mathematical graph using an adjacency matrix.
 *
 * \tparam Matrix is type of matrix used to store the adjacency matrix of the graph.
 * \tparam GraphType is type of the graph - directed or undirected.
 *
 * \par Examples
 * See \ref GraphExample_Constructors.cpp, \ref GraphExample_setEdges.cpp, and \ref GraphExample_VertexView.cpp for usage
 * examples.
 */
template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
struct GraphBase
{
   static_assert( Matrices::is_matrix_v< AdjacencyMatrix > && ! Matrices::is_matrix_view_v< AdjacencyMatrix >,
                  "Adjacency matrix type cannot be matrix view." );

   //! \brief Type of the adjacency matrix view.
   using AdjacencyMatrixView = typename AdjacencyMatrix::ViewType;

   //! \brief Type of constant view of the adjacency matrix.
   using ConstAdjacencyMatrixView = typename AdjacencyMatrix::ConstViewType;

   //! \brief Type for indexing of the graph nodes.
   using IndexType = Index;

   //! \brief Type of device where the graph will be operating.
   using DeviceType = Device;

   //! \brief Type for weights of the graph edges.
   using ValueType = Value;

   //! \brief Type of the graph - directed or undirected.
   using GraphOrientation = Orientation;

   using VertexView = GraphVertexView< AdjacencyMatrixView, Orientation >;

   using ConstVertexView = typename VertexView::ConstVertexView;

   template< typename Value_ = Value,
             typename Device_ = Device,
             typename Index_ = Index,
             typename Orientation_ = Orientation,
             typename AdjacencyMatrix_ = AdjacencyMatrix >
   using Self = GraphBase< Value_, Device_, Index_, Orientation_, AdjacencyMatrix_ >;

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

   //! \brief Returns the constant view of adjacency matrix of the graph.
   [[nodiscard]] __cuda_callable__
   const AdjacencyMatrixView&
   getAdjacencyMatrixView() const;

   //! \brief Returns the view of adjacency matrix of the graph.
   [[nodiscard]] __cuda_callable__
   AdjacencyMatrixView&
   getAdjacencyMatrixView();

   //! \brief Returns the number of nodes in the graph.
   [[nodiscard]] __cuda_callable__
   IndexType
   getVertexCount() const;

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
   ConstVertexView
   getVertex( IndexType vertexIdx ) const;

   /***
    * \brief Returns the modifiable view of the graph node with given index.
    *
    * \param nodeIdx is index of the node to be returned.
    */
   [[nodiscard]] __cuda_callable__
   VertexView
   getVertex( IndexType vertexIdx );

   /***
    * \brief Sets the weight of the edge between given node and its edge index.
    *
    * \param nodeIdx is index of the node.
    * \param edgeIdx is index of the edge of the node.
    * \param value is new weight of the edge.
    */
   __cuda_callable__
   void
   setEdgeWeight( IndexType vertexIdx, IndexType edgeIdx, const ValueType& value );

   // TODO: Add eraseEdge - works only for sparse adjacency matrices
   /***
    * \brief Returns the weight of the edge between given node and its edge index.
    *
    * \param nodeIdx is index of the node.
    * \param edgeIdx is index of the edge of the node.
    */
   [[nodiscard]] __cuda_callable__
   ValueType
   getEdgeWeight( IndexType vertexIdx, IndexType edgeIdx ) const;

   [[nodiscard]] __cuda_callable__
   IndexType
   getVertexDegree( IndexType nodeIdx ) const
   {
      return this->getVertex( nodeIdx ).getDegree();
   }

   //! \brief Destructor.
   ~GraphBase() = default;

protected:
   AdjacencyMatrixView adjacencyMatrixView;
};

//! \brief Output stream operator for the \e Graph class.
template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
std::ostream&
operator<<( std::ostream& os, const GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >& graph );

//! \brief Serialization of graphs into binary files.
template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
File&
operator<<( File& file, const GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >& graph );

//! \brief Serialization of graphs into binary files.
template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
File&
operator<<( File&& file, const GraphBase< Value, Device, Index, Orientation, AdjacencyMatrix >& graph );

// Note: Deserialization is different for Graph and GraphView,
// see the respective files for implementation.

}  // namespace TNL::Graphs

#include "GraphBase.hpp"
