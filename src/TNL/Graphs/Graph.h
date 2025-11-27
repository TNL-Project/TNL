// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <ostream>
#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Matrices/TypeTraits.h>

namespace TNL::Graphs {

//! Enumeration of supported graph types - directed or undirected.
enum class GraphTypes : std::uint8_t
{
   Directed,   //!< Directed graphs.
   Undirected  //!< Undirected graphs.
};

/**
 * \brief \e Graph class represents a mathematical graph using an adjacency matrix.
 *
 * \tparam Matrix is type of matrix used to store the adjacency matrix of the graph.
 * \tparam GraphType is type of the graph - directed or undirected.
 */
template< typename Matrix, GraphTypes GraphType = GraphTypes::Directed >
struct Graph
{
   static_assert( Matrices::is_matrix_v< Matrix > );

   //! \brief Type of the adjacency matrix.
   using MatrixType = Matrix;

   //! \brief Type for indexing of the graph nodes.
   using IndexType = typename Matrix::IndexType;

   //! \brief Type of device where the graph will be operating.
   using DeviceType = typename Matrix::DeviceType;

   //! \brief Type for weights of the graph edges.
   using ValueType = typename Matrix::RealType;

   //! \brief Checks if the graph is directed.
   static constexpr bool
   isDirected();

   //! \brief Checks if the graph is undirected.
   static constexpr bool
   isUndirected();

   //! \brief Returns the type of the graph.
   static constexpr GraphTypes
   getGraphType();

   //! \brief Default constructor.
   Graph() = default;

   //! \brief Constructor with adjacency matrix.
   Graph( const MatrixType& matrix );

   //! \brief Constructor with adjacency matrix.
   Graph( MatrixType&& matrix );

   //! \brief Copy constructor.
   Graph( const Graph& ) = default;

   //! \brief Move constructor.
   Graph( Graph&& ) = default;

   //! \brief Templated copy constructor from another graph type.
   template< typename OtherGraph >
   Graph( const OtherGraph& other );

   //! \brief Templated move constructor from another graph type.
   template< typename OtherGraph >
   Graph( const OtherGraph&& other );

   /**
    * \brief Constructor with number of nodes and edges given as initializer list.
    *
    * \param nodesCount is the number of nodes in the graph.
    * \param data is the initializer list of tuples (source node, target node, edge weight).
    * \param encoding is the encoding for symmetric matrices (used only for undirected graphs).
    *
    * If the graph is undirected, the adjacency matrix can be symmetric. In this case, the
    * parameter \a encoding specifies whether the lower or upper part of the matrix is provided.
    * If the graph is undirected and the adjacency matrix type is not symmetric, the constructor
    * will create a symmetric adjacency matrix by adding both (source, target) and (target, source)
    * entries for each edge.
    */
   Graph( IndexType nodesCount,
          const std::initializer_list< std::tuple< IndexType, IndexType, ValueType > >& data,
          Matrices::SymmetricMatrixEncoding encoding = Matrices::SymmetricMatrixEncoding::LowerPart );

   /**
    * \brief Constructor with number of nodes and edges given as a map.
    *
    * \param nodesCount is the number of nodes in the graph.
    * \param map is the map with keys as (source node, target node) pairs and values as edge weights.
    * \param encoding is the encoding for symmetric matrices (used only for undirected graphs).
    *
    * If the graph is undirected, the adjacency matrix can be symmetric. In this case, the
    * parameter \a encoding specifies whether the lower or upper part of the matrix is provided.
    * If the graph is undirected and the adjacency matrix type is not symmetric, the constructor
    * will create a symmetric adjacency matrix by adding both (source, target) and (target, source)
    * entries for each edge.
    */
   template< typename MapIndex, typename MapValue >
   Graph( IndexType nodesCount,
          const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
          Matrices::SymmetricMatrixEncoding encoding = Matrices::SymmetricMatrixEncoding::LowerPart );

   //! \brief Copy-assignment operator.
   Graph&
   operator=( const Graph& ) = default;

   //! \brief Move-assignment operator.
   Graph&
   operator=( Graph&& ) = default;

   //! \brief Comparisons operator.
   bool
   operator==( const Graph& other ) const;

   //! \brief Sets the number of nodes in the graph.
   void
   setNodeCount( IndexType nodesCount );

   /**
    * \brief Sets the edges of the graph from a map.
    *
    * \tparam MapIndex is type for indexing of the nodes in the map.
    * \tparam MapValue is type for weights of the edges in the map.
    *
    * \param map is the map with keys as (source node, target node) pairs and values as edge weights.
    */
   template< typename MapIndex, typename MapValue >
   void
   setEdges( const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map );

   //! \brief Returns the number of nodes in the graph.
   [[nodiscard]] IndexType
   getNodeCount() const;

   //! \brief Returns the number of edges in the graph.
   [[nodiscard]] IndexType
   getEdgeCount() const;

   /**
    * \brief Sets the capacities of the graph nodes.
    *
    * \param nodeCapacities is the vector holding the node capacities.
    *
    * The method sets the row capacities of the adjacency matrix to the provided node capacities.
    * It is not necessary to call this method if the adjacency matrix is dense. If the adjacency
    * matrix is sparse and symmetric, capacity only for edges in one direction (e.g., lower part)
    * should be provided.
    */
   template< typename Vector >
   void
   setNodeCapacities( const Vector& nodeCapacities );

   //! \brief Returns the constant adjacency matrix of the graph.
   [[nodiscard]] const MatrixType&
   getAdjacencyMatrix() const;

   //! \brief Returns the modifiable adjacency matrix of the graph.
   MatrixType&
   getAdjacencyMatrix();

   /**
    * \brief Sets the adjacency matrix of the graph.
    *
    * \tparam Matrix_ is type of the input adjacency matrix.
    *
    * \param matrix is the input adjacency matrix.
    *
    * The method makes a copy of the provided adjacency matrix and so
    * the type if the input matrix can be different from the type of the
    * adjacency matrix of the graph.
    */
   template< typename Matrix_ >
   void
   setAdjacencyMatrix( Matrix_ matrix );

   //! \brief Destructor.
   ~Graph() = default;

protected:
   MatrixType adjacencyMatrix;
};

//! \brief Output stream operator for the \e Graph class.
template< typename Matrix, GraphTypes GraphType >
std::ostream&
operator<<( std::ostream& os, const Graph< Matrix, GraphType >& graph );

}  // namespace TNL::Graphs

#include "Graph.hpp"
