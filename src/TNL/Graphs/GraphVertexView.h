// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseMatrixRowView.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseMatrixRowView.h>

namespace TNL::Graphs {

template< typename Value,
          typename Device,
          typename Index,
          typename Orientation,
          template< typename, typename, typename > class Segments,
          typename AdjacencyMatrix >
struct Graph;

/**
 * \brief View type for accessing individual graph vertices and their edges.
 *
 * GraphVertexView provides access to a single vertex (node) in a graph, allowing iteration over
 * its edges, accessing edge weights, and modifying edge properties. It is similar to a matrix row view
 * but provides graph-specific interface methods.
 *
 * The GraphVertexView is typically obtained from a Graph using methods like `getVertex()` or during
 * parallel traversal with `forAllVertices()`. It wraps either a dense or sparse matrix row view,
 * depending on the underlying adjacency matrix type.
 *
 * \tparam MatrixView Type of the underlying matrix view (DenseMatrixView or SparseMatrixView).
 * \tparam Orientation Graph orientation (DirectedGraph or UndirectedGraph).
 *
 * \par Example
 * \include GraphExample_VertexView.cpp
 * \par Output
 * \include GraphExample_VertexView.out
 */
template< typename MatrixView, typename Orientation >
struct GraphVertexView;

/**
 * \brief Specialization of GraphVertexView for sparse adjacency matrices.
 *
 * This specialization is used when the graph stores its adjacency information using a sparse matrix.
 * In this case, only the actual edges are stored, and edge indices are separate from target vertex indices.
 * The sparse representation is more memory-efficient for graphs with few edges per vertex.
 *
 * \tparam Real Type of edge weights.
 * \tparam Device Device type (Host or Cuda).
 * \tparam Index Type for indexing vertices and edges.
 * \tparam MatrixType_ Matrix type for the sparse matrix.
 * \tparam SegmentsView Segments view type for the sparse matrix.
 * \tparam ComputeRealType Computation real type.
 * \tparam GraphType_ Graph type for orientation information.
 */
template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType_,
          template< typename Device_, typename Index_ > class SegmentsView,
          typename ComputeRealType,
          typename GraphType_ >
struct GraphVertexView< Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >,
                        GraphType_ >
: public Matrices::SparseMatrixRowView<
     typename Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >::SegmentsViewType::
        SegmentViewType,
     typename Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >::ValuesViewType,
     typename Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >::
        ColumnIndexesViewType >
{
   //! \brief Type of the sparse matrix view.
   using MatrixView = Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >;
   //! \brief Type of constant sparse matrix view.
   using ConstMatrixView = typename MatrixView::ConstViewType;

   using MatrixRowView = Matrices::SparseMatrixRowView< typename MatrixView::SegmentsViewType::SegmentViewType,
                                                        typename MatrixView::ValuesViewType,
                                                        typename MatrixView::ColumnIndexesViewType >;

   using ConstMatrixRowView = typename MatrixRowView::ConstRowView;

   //! \brief Base type.
   using Base = Matrices::SparseMatrixRowView<
      typename Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >::SegmentsViewType::
         SegmentViewType,
      typename Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >::ValuesViewType,
      typename Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >::
         ColumnIndexesViewType >;

   using ColumnIndexesViewType = typename Base::ColumnsIndexesViewType;  // TODO: Rename ColumnsIndexesViewType in Base

   using typename Base::IndexType;
   using typename Base::RealType;
   using typename Base::SegmentViewType;
   using typename Base::ValuesViewType;

   //! \brief Type of the graph node view.
   using VertexView = GraphVertexView< MatrixView, GraphType_ >;

   //! \brief Type of the constant graph node view.
   using ConstVertexView = GraphVertexView< ConstMatrixView, GraphType_ >;

   /**
    * \brief Constructs a vertex view from segment, values, and column index views.
    *
    * \param segmentView View of the matrix segment (row metadata).
    * \param valuesView View of edge weights.
    * \param columnIndexesView View of target vertex indices.
    */
   __cuda_callable__
   GraphVertexView( const SegmentViewType& segmentView,
                    const ValuesViewType& valuesView,
                    const ColumnIndexesViewType& columnIndexesView )
   : Base( segmentView, valuesView, columnIndexesView )
   {}

   /**
    * \brief Constructs a vertex view by moving from a matrix row view.
    *
    * \param matrixRowView Matrix row view to move from.
    */
   __cuda_callable__
   GraphVertexView( MatrixRowView&& matrixRowView )
   : Base( std::forward< MatrixRowView >( matrixRowView ) )
   {}

   /**
    * \brief Constructs a vertex view from a matrix row view.
    *
    * \param matrixRowView Matrix row view to copy from.
    */
   __cuda_callable__
   GraphVertexView( const MatrixRowView& matrixRowView )
   : Base( matrixRowView )
   {}

   /**
    * \brief Copy constructor.
    */
   __cuda_callable__
   GraphVertexView( const GraphVertexView& ) = default;

   /**
    * \brief Returns the index of this vertex in the graph.
    *
    * \return The vertex index.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getVertexIndex() const
   {
      return Base::getRowIndex();
   }

   /**
    * \brief Returns the weight of the edge at the given edge index (const version).
    *
    * \param edgeIndex Index of the edge (0 to getDegree()-1).
    * \return Const reference to the edge weight.
    */
   [[nodiscard]] __cuda_callable__
   const RealType&
   getEdgeWeight( IndexType edgeIndex ) const
   {
      return Base::getValue( edgeIndex );
   }

   /**
    * \brief Returns the weight of the edge at the given edge index (non-const version).
    *
    * \param edgeIndex Index of the edge (0 to getDegree()-1).
    * \return Reference to the edge weight.
    */
   [[nodiscard]] __cuda_callable__
   RealType&
   getEdgeWeight( IndexType edgeIndex )
   {
      return Base::getValue( edgeIndex );
   }

   /**
    * \brief Returns the target vertex index of the edge at the given edge index.
    *
    * \param edgeIndex Index of the edge (0 to getDegree()-1).
    * \return Index of the target vertex.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getTargetIndex( IndexType edgeIndex ) const
   {
      return Base::getColumnIndex( edgeIndex );
   }

   /**
    * \brief Sets the weight of the edge at the given edge index.
    *
    * \param edgeIndex Index of the edge (0 to getDegree()-1).
    * \param weight New weight for the edge.
    */
   __cuda_callable__
   void
   setEdgeWeight( IndexType edgeIndex, const RealType& weight )
   {
      Base::setValue( edgeIndex, weight );
   }

   /**
    * \brief Sets both the target vertex and weight for an edge.
    *
    * For sparse matrices, this method updates both the target vertex index and the edge weight
    * at the specified edge index.
    *
    * \param edgeIndex Index of the edge (0 to getDegree()-1).
    * \param target New target vertex index.
    * \param weight New weight for the edge.
    */
   __cuda_callable__
   void
   setEdge( IndexType edgeIndex, IndexType target, const RealType& weight )
   {
      Base::setColumnIndex( edgeIndex, target );
      Base::setValue( edgeIndex, weight );
   }

   /**
    * \brief Returns the degree (number of edges) of this vertex.
    *
    * For sparse matrices, this counts only the valid edges (excluding padding entries).
    * Padding entries are marked with paddingIndex and are not counted as real edges.
    *
    * \return The vertex degree.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getDegree() const
   {
      IndexType degree = 0;
      for( IndexType i = 0; i < Base::getSize(); ++i )
         if( Base::getColumnIndex( i ) != Matrices::paddingIndex< IndexType > )
            ++degree;
      return degree;
   }
};

/**
 * \brief Specialization of GraphVertexView for dense adjacency matrices.
 *
 * This specialization is used when the graph stores its adjacency information using a dense matrix.
 * In this case, edge indices correspond directly to target vertex indices, and all possible edges
 * (for the vertex degree) are stored explicitly.
 *
 * \tparam Real Type of edge weights.
 * \tparam Device Device type (Host or Cuda).
 * \tparam Index Type for indexing vertices and edges.
 * \tparam Organization Dense matrix elements organization.
 * \tparam GraphType_ Graph type for orientation information.
 */
template< typename Real,
          typename Device,
          typename Index,
          TNL::Algorithms::Segments::ElementsOrganization Organization,
          typename GraphType_ >
struct GraphVertexView< Matrices::DenseMatrixView< Real, Device, Index, Organization >, GraphType_ >
: public Matrices::DenseMatrixRowView< typename Matrices::DenseMatrixView< Real, Device, Index, Organization >::SegmentViewType,
                                       typename Matrices::DenseMatrixView< Real, Device, Index, Organization >::ValuesViewType >
{
   //! \brief Type of the dense matrix view.
   using MatrixView = Matrices::DenseMatrixView< Real, Device, Index, Organization >;

   //! \brief Type of constant dense matrix view.
   using ConstMatrixView = typename MatrixView::ConstViewType;

   using MatrixRowView =
      Matrices::DenseMatrixRowView< typename MatrixView::SegmentViewType, typename MatrixView::ValuesViewType >;

   using ConstMatrixRowView = typename MatrixRowView::ConstRowView;

   //! \brief Base type.
   using Base = Matrices::DenseMatrixRowView< typename MatrixView::SegmentViewType, typename MatrixView::ValuesViewType >;

   using typename Base::IndexType;
   using typename Base::RealType;
   using typename Base::SegmentViewType;
   using typename Base::ValuesViewType;

   //! \brief Type of the graph node view.
   using VertexView = GraphVertexView< MatrixView, GraphType_ >;

   //! \brief Type of the constant graph node view.
   using ConstVertexView = GraphVertexView< ConstMatrixView, GraphType_ >;

   /**
    * \brief Constructs a vertex view from segment and values views.
    *
    * \param segmentView View of the matrix segment (row metadata).
    * \param valuesView View of edge weights.
    */
   __cuda_callable__
   GraphVertexView( const SegmentViewType& segmentView, const ValuesViewType& valuesView )
   : Base( segmentView, valuesView )
   {}

   /**
    * \brief Constructs a vertex view by moving from a matrix row view.
    *
    * \param matrixRowView Matrix row view to move from.
    */
   __cuda_callable__
   GraphVertexView( MatrixRowView&& matrixRowView )
   : Base( std::forward< MatrixRowView >( matrixRowView ) )
   {}

   /**
    * \brief Constructs a vertex view from a matrix row view.
    *
    * \param matrixRowView Matrix row view to copy from.
    */
   __cuda_callable__
   GraphVertexView( const MatrixRowView& matrixRowView )
   : Base( matrixRowView )
   {}

   /**
    * \brief Copy constructor.
    */
   __cuda_callable__
   GraphVertexView( const GraphVertexView& ) = default;

   /**
    * \brief Returns the index of this vertex in the graph.
    *
    * \return The vertex index.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getVertexIndex() const
   {
      return Base::getRowIndex();
   }

   /**
    * \brief Returns the weight of the edge at the given edge index (const version).
    *
    * \param edgeIndex Index of the edge (0 to getDegree()-1).
    * \return Const reference to the edge weight.
    */
   [[nodiscard]] __cuda_callable__
   const RealType&
   getEdgeWeight( IndexType edgeIndex ) const
   {
      return Base::getValue( edgeIndex );
   }

   /**
    * \brief Returns the weight of the edge at the given edge index (non-const version).
    *
    * \param edgeIndex Index of the edge (0 to getDegree()-1).
    * \return Reference to the edge weight.
    */
   [[nodiscard]] __cuda_callable__
   RealType&
   getEdgeWeight( IndexType edgeIndex )
   {
      return Base::getValue( edgeIndex );
   }

   /**
    * \brief Returns the target vertex index of the edge at the given edge index.
    *
    * \param edgeIndex Index of the edge (0 to getDegree()-1).
    * \return Index of the target vertex.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getTargetIndex( IndexType edgeIndex ) const
   {
      return Base::getColumnIndex( edgeIndex );
   }

   /**
    * \brief Sets the weight of the edge at the given edge index.
    *
    * \param edgeIndex Index of the edge (0 to getDegree()-1).
    * \param weight New weight for the edge.
    */
   __cuda_callable__
   void
   setEdgeWeight( IndexType edgeIndex, const RealType& weight )
   {
      Base::setValue( edgeIndex, weight );
   }

   /**
    * \brief Sets both the target vertex and weight for an edge.
    *
    * For dense matrices, the edge is identified by its target index, so the edgeIndex parameter
    * is not used. This method sets the weight for the edge to the specified target vertex.
    *
    * \param edgeIndex Edge index (it is ignored for dense adjacency matrices).
    * \param target Target vertex index.
    * \param weight New weight for the edge.
    */
   __cuda_callable__
   void
   setEdge( IndexType edgeIndex, IndexType target, const RealType& weight )
   {
      // In dense matrix, edges are identified by their target indices
      Base::setValue( target, weight );
   }

   /**
    * \brief Returns the degree (number of edges) of this vertex.
    *
    * For dense matrices, this is the size of the row, which corresponds to the number of vertices
    * in the graph.
    *
    * \return The vertex degree.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getDegree() const
   {
      return Base::getSize();
   }
};

}  // namespace TNL::Graphs
