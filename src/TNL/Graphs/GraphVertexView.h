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

template< typename MatrixView, typename Orientation >
struct GraphVertexView;

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

   __cuda_callable__
   GraphVertexView( const SegmentViewType& segmentView, const ValuesViewType& valuesView )
   : Base( segmentView, valuesView )
   {}

   __cuda_callable__
   GraphVertexView( MatrixRowView&& matrixRowView )
   : Base( std::forward( matrixRowView ) )
   {}

   __cuda_callable__
   GraphVertexView( const MatrixRowView& matrixRowView )
   : Base( matrixRowView )
   {}

   __cuda_callable__
   GraphVertexView( const GraphVertexView& ) = default;

   [[nodiscard]] __cuda_callable__
   IndexType
   getVertexIndex() const
   {
      return Base::getRowIndex();
   }

   [[nodiscard]] __cuda_callable__
   const RealType&
   getEdgeWeight( IndexType edgeIndex ) const
   {
      return Base::getValue( edgeIndex );
   }

   [[nodiscard]] __cuda_callable__
   RealType&
   getEdgeWeight( IndexType edgeIndex )
   {
      return Base::getValue( edgeIndex );
   }

   [[nodiscard]] __cuda_callable__
   IndexType
   getTargetIndex( IndexType edgeIndex ) const
   {
      return Base::getColumnIndex( edgeIndex );
   }

   __cuda_callable__
   void
   setEdgeWeight( IndexType edgeIndex, const RealType& weight )
   {
      Base::setValue( edgeIndex, weight );
   }

   __cuda_callable__
   void
   setEdge( IndexType edgeIndex, IndexType target, const RealType& weight )
   {
      // In dense matrix, edges are identified by their target indices
      Base::setValue( target, weight );
   }

   [[nodiscard]] __cuda_callable__
   IndexType
   getDegree() const
   {
      return Base::getSize();
   }
};

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
   using Base = Matrices::SparseMatrixRowView< typename MatrixView::SegmentsViewType::SegmentViewType,
                                               typename MatrixView::ValuesViewType,
                                               typename MatrixView::ColumnIndexesViewType >;

   using ColumnIndexesViewType = typename Base::ColumnsIndexesViewType;  // TODO: Rename ColumnsIndexesViewType in Base

   using typename Base::IndexType;
   using typename Base::RealType;
   using typename Base::SegmentViewType;
   using typename Base::ValuesViewType;

   //! \brief Type of the graph node view.
   using VertexView = GraphVertexView< MatrixView, GraphType_ >;

   //! \brief Type of the constant graph node view.
   using ConstVertexView = GraphVertexView< ConstMatrixView, GraphType_ >;

   __cuda_callable__
   GraphVertexView( const SegmentViewType& segmentView,
                    const ValuesViewType& valuesView,
                    const ColumnIndexesViewType& columnIndexesView )
   : Base( segmentView, valuesView, columnIndexesView )
   {}

   __cuda_callable__
   GraphVertexView( MatrixRowView&& matrixRowView )
   : Base( std::forward( matrixRowView ) )
   {}

   __cuda_callable__
   GraphVertexView( const MatrixRowView& matrixRowView )
   : Base( matrixRowView )
   {}

   __cuda_callable__
   GraphVertexView( const GraphVertexView& ) = default;

   [[nodiscard]] __cuda_callable__
   IndexType
   getVertexIndex() const
   {
      return Base::getRowIndex();
   }

   [[nodiscard]] __cuda_callable__
   const RealType&
   getEdgeWeight( IndexType edgeIndex ) const
   {
      return Base::getValue( edgeIndex );
   }

   [[nodiscard]] __cuda_callable__
   RealType&
   getEdgeWeight( IndexType edgeIndex )
   {
      return Base::getValue( edgeIndex );
   }

   [[nodiscard]] __cuda_callable__
   IndexType
   getTargetIndex( IndexType edgeIndex ) const
   {
      return Base::getColumnIndex( edgeIndex );
   }

   __cuda_callable__
   void
   setEdgeWeight( IndexType edgeIndex, const RealType& weight )
   {
      Base::setValue( edgeIndex, weight );
   }

   __cuda_callable__
   void
   setEdge( IndexType edgeIndex, IndexType target, const RealType& weight )
   {
      Base::setColumnIndex( edgeIndex, target );
      Base::setValue( edgeIndex, weight );
   }

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

}  // namespace TNL::Graphs
