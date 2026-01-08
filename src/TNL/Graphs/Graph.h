// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <ostream>
#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Matrices/TypeTraits.h>

#include "GraphBase.h"
#include "GraphView.h"
#include "GraphVertexView.h"
#include "TypeTraits.h"
#include "GraphOrientation.h"
#include "detail/DefaultGraphAdjacencyMatrixType.h"

namespace TNL::Graphs {

/**
 * \brief \e Graph class represents a mathematical graph using an adjacency matrix.
 *
 * By default, the adjacency matrix is stored using a sparse matrix representation. This behavior can be changed by specifying
 * adifferent matrix type as the last template parameter.
 *
 * When a sparse matrix is used, all matrix elements that are not explicitly stored are interpreted as zero. In the context of
 * graphs, however, it is important to distinguish between the absence of an edge and the presence of an edge with zero weight.
 * To preserve this distinction, edges with zero weight are represented by explicitly stored matrix entries with zero value.
 *
 * For unweighted graphs, use `bool` as the `Value` type. In this case, the sparse adjacency matrix is binary, which only
 * stores the positions of the edges without any associated weights.
 *
 * If a dense matrix is used as the adjacency matrix, it represents a complete graph, meaning that all possible edges between
 * nodes are present.
 *
 * \tparam Value is type for weights of the graph edges.
 * \tparam Device is type of device where the graph will be operating.
 * \tparam Index is type for indexing of the graph nodes.
 * \tparam Orientation is type of the graph - directed or undirected.
 * \tparam AdjacencyMatrix is type of matrix used to store the adjacency matrix of the graph.
 */
template< typename Value,
          typename Device,
          typename Index,
          typename Orientation = DirectedGraph,
          template< typename, typename, typename > class Segments = TNL::Algorithms::Segments::CSR,
          typename AdjacencyMatrix = detail::DefaultGraphAdjacencyMatrix_t< Value, Device, Index, Segments, Orientation > >
struct Graph
: public GraphBase< Value,
                    Device,
                    Index,
                    Orientation,
                    std::conditional_t< std::is_const_v< Value >, typename AdjacencyMatrix::ConstMatrixType, AdjacencyMatrix > >
{
   //! \brief Type of the adjacency matrix.
   using AdjacencyMatrixType =
      std::conditional_t< std::is_const_v< Value >, typename AdjacencyMatrix::ConstMatrixType, AdjacencyMatrix >;

   //! \brief Type of the adjacency matrix view.
   using AdjacencyMatrixView = typename AdjacencyMatrixType::ViewType;

   //! \brief Type of constant view of the adjacency matrix.
   using ConstAdjacencyMatrixView = typename AdjacencyMatrixType::ConstViewType;

   //! \brief Type for weights of the graph edges.
   using ValueType = std::remove_cv_t< Value >;

   //! \brief Type for indexing of the graph nodes.
   using IndexType = Index;

   //! \brief Type of device where the graph will be operating.
   using DeviceType = Device;

   //! \brief Type of the graph - directed or undirected.
   using GraphOrientation = Orientation;

   //! \brief Type of constant view of the adjacency matrix.
   using ViewType = GraphView< Value, Device, Index, Orientation, AdjacencyMatrixType >;

   //! \brief Type of constant view of the adjacency matrix.
   using ConstViewType =
      GraphView< std::add_const_t< Value >, Device, Index, Orientation, typename AdjacencyMatrixType::ConstMatrixType >;

   //! \brief Type of the graph nodes view.
   using VertexView = GraphVertexView< AdjacencyMatrixView, Orientation >;

   //! \brief Type of constant graph nodes view.
   using ConstVertexView = typename VertexView::ConstVertexView;

   //! \brief Helper type for getting self type or its modifications.
   template< typename Value_ = Value,
             typename Device_ = Device,
             typename Index_ = Index,
             typename Orientation_ = Orientation,
             template< typename, typename, typename > class Segments_ = Segments,
             typename AdjacencyMatrix_ = AdjacencyMatrix >
   using Self = Graph< Value_, Device_, Index_, Orientation_, Segments_, AdjacencyMatrix_ >;

   using Base = GraphBase< Value, Device, Index, Orientation, AdjacencyMatrixType >;
   using Base::isDirected;
   using Base::isUndirected;

   //! \brief Default constructor.
   Graph() = default;

   //! \brief Constructor with number of nodes.
   Graph( IndexType nodesCount );

   //! \brief Constructor with adjacency matrix.
   Graph( const AdjacencyMatrixType& matrix );

   //! \brief Constructor with adjacency matrix.
   Graph( AdjacencyMatrixType&& matrix );

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
    * \param vertexCount is the number of nodes in the graph.
    * \param data is the initializer list of tuples (source node, target node, edge weight).
    * \param encoding is the encoding for symmetric matrices (used only for undirected graphs).
    *
    * If the graph is undirected, the adjacency matrix can be symmetric. In this case, the
    * parameter \a encoding specifies whether the lower or upper part of the matrix is provided.
    * If the graph is undirected and the adjacency matrix type is not symmetric, the constructor
    * will create a symmetric adjacency matrix by adding both (source, target) and (target, source)
    * entries for each edge.
    */
   Graph( IndexType vertexCount,
          const std::initializer_list< std::tuple< IndexType, IndexType, ValueType > >& data,
          Matrices::MatrixElementsEncoding encoding = isDirected() ? Matrices::MatrixElementsEncoding::Complete
                                                                   : Matrices::MatrixElementsEncoding::SymmetricMixed );

   /**
    * \brief Constructor with number of nodes and edges given as a map.
    *
    * \param vertexCount is the number of nodes in the graph.
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
   Graph( IndexType vertexCount,
          const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
          Matrices::MatrixElementsEncoding encoding = isDirected() ? Matrices::MatrixElementsEncoding::Complete
                                                                   : Matrices::MatrixElementsEncoding::SymmetricMixed );

   //! \brief Copy-assignment operator.
   Graph&
   operator=( const Graph& other );

   template< typename OtherGraph, std::enable_if_t< isGraph< OtherGraph >( std::declval< OtherGraph >() ) > >
   Graph&
   operator=( const OtherGraph& other );

   //! \brief Move-assignment operator.
   Graph&
   operator=( Graph&& other );

   //! \brief Returns the modifiable view of the graph.
   ViewType
   getView();

   //! \brief Returns the constant view of the graph.
   ConstViewType
   getConstView() const;

   //! \brief Sets the number of nodes in the graph.
   void
   setVertexCount( IndexType nodesCount );

   /**
    * \brief Sets the edge counts (capacities) for all vertices in the graph.
    *
    * \tparam Vector is type of the vector holding the edge counts.
    *
    * \param edgeCounts is the vector holding the number of edges for each vertex.
    *
    * This method sets the row capacities of the adjacency matrix to the provided edge counts.
    * For undirected graphs with symmetric adjacency matrix, only capacities for edges in one
    * direction should be provided.
    */
   template< typename Vector >
   void
   setEdgeCounts( const Vector& edgeCounts );

   /**
    * \brief Sets the edges of the graph from an initializer list.
    *
    * The edge values are given as a list \e data of triples:
    * { { source1, target1, weight1 },
    *   { source2, target2, weight2 },
    * ... }.
    *
    * \param data is an initializer list of tuples representing edges (source, target, weight).
    * \param encoding defines encoding for symmetric matrices (used only for undirected graphs).
    *
    * See \ref TNL::Matrices::SparseMatrix::setElements for details on how the \e encoding parameter works.
    */
   void
   setEdges( const std::initializer_list< std::tuple< IndexType, IndexType, ValueType > >& data,
             Matrices::MatrixElementsEncoding encoding = isDirected() ? Matrices::MatrixElementsEncoding::Complete
                                                                      : Matrices::MatrixElementsEncoding::SymmetricMixed );

   /**
    * \brief Sets the edges of the graph from a map.
    *
    * \tparam MapIndex is type for indexing of the nodes in the map.
    * \tparam MapValue is type for weights of the edges in the map.
    *

    * \param map is the map with keys as (source node, target node) pairs and values as edge weights.
    * \param encoding defines encoding for symmetric matrices (used only for undirected graphs).
    *
    * See \ref TNL::Matrices::SparseMatrix::setElements for details on how the \e encoding parameter works.
    */
   template< typename MapIndex, typename MapValue >
   void
   setEdges( const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
             Matrices::MatrixElementsEncoding encoding = isDirected() ? Matrices::MatrixElementsEncoding::Complete
                                                                      : Matrices::MatrixElementsEncoding::SymmetricMixed );

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
   setVertexCapacities( const Vector& nodeCapacities );

   //! \brief Returns the modifiable adjacency matrix of the graph.
   [[nodiscard]] const AdjacencyMatrixType&
   getAdjacencyMatrix() const;

   //! \brief Returns the modifiable adjacency matrix of the graph.
   [[nodiscard]] AdjacencyMatrixType&
   getAdjacencyMatrix();

   //! \brief Sets the adjacency matrix of the graph.
   void
   setAdjacencyMatrix( const AdjacencyMatrixType& matrix );

   //! \brief Sets the adjacency matrix of the graph.
   void
   setAdjacencyMatrix( AdjacencyMatrixType&& matrix );
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
   setAdjacencyMatrix( const Matrix_& matrix );

   /**
    * \brief Resets the graph to zero vertices and edges.
    *
    * This method resets the adjacency matrix to zero dimensions, effectively
    * removing all vertices and edges from the graph.
    */
   void
   reset();

   //! \brief Destructor.
   ~Graph() = default;

protected:
   AdjacencyMatrixType adjacencyMatrix;
};

//! \brief Deserialization of graphs from binary files.
template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
File&
operator>>( File& file, Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >& graph );

//! \brief Deserialization of graphs from binary files.
template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
File&
operator>>( File&& file, Graph< Value, Device, Index, Orientation, Segments, AdjacencyMatrix >& graph );

}  // namespace TNL::Graphs

#include "Graph.hpp"
