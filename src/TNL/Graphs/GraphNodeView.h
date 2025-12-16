// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseMatrixRowView.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseMatrixRowView.h>

namespace TNL::Graphs {

template< typename MatrixView, typename GraphType_ >
struct Graph;

template< typename MatrixView, typename GraphType_ >
struct GraphNodeView;

template< typename Real,
          typename Device,
          typename Index,
          Algorithms::Segments::ElementsOrganization Organization,
          typename GraphType_ >
struct GraphNodeView< Matrices::DenseMatrixView< Real, Device, Index, Organization >, GraphType_ >
: public Matrices::DenseMatrixRowView<
     typename Matrices::DenseMatrixView< Real, Device, Index, Organization >::SegmentsViewType,
     typename Matrices::DenseMatrixView< Real, Device, Index, Organization >::ValuesViewType >
{
   //! \brief Type of the dense matrix view.
   using MatrixView = Matrices::DenseMatrixView< Real, Device, Index, Organization >;

   //! \brief Type of constant dense matrix view.
   using ConstMatrixView = typename MatrixView::ConstViewType;

   //! \brief Base type.
   using Base = Matrices::DenseMatrixRowView< typename MatrixView::SegmentsViewType, typename MatrixView::ValuesViewType >;

   using typename Base::IndexType;
   using typename Base::RealType;
   using typename Base::SegmentViewType;
   using typename Base::ValuesViewType;

   //! \brief Type of the graph node view.
   using NodeView = GraphNodeView< MatrixView, GraphType_ >;

   //! \brief Type of the constant graph node view.
   using ConstNodeView = GraphNodeView< ConstMatrixView, GraphType_ >;

   GraphNodeView( const SegmentViewType& segmentView, const ValuesViewType& valuesView )
   : Base( segmentView, valuesView )
   {}

   [[nodiscard]] __cuda_callable__
   IndexType
   getNodeIndex() const
   {
      return Base::getRowIndex();
   }

   [[nodiscard]] __cuda_callable__
   const RealType&
   getWeight( IndexType edgeIndex ) const
   {
      return Base::getValue( edgeIndex );
   }

   [[nodiscard]] __cuda_callable__
   RealType&
   getWeight( IndexType edgeIndex )
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
   setWeight( IndexType edgeIndex, const RealType& weight )
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
};

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType_,
          template< typename Device_, typename Index_ > class SegmentsView,
          typename ComputeRealType,
          typename GraphType_ >
struct GraphNodeView< Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >,
                      GraphType_ >
: public Matrices::SparseMatrixRowView<
     typename Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >::SegmentsViewType,
     typename Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >::ValuesViewType,
     typename Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >::
        ColumnIndexesViewType >
{
   //! \brief Type of the sparse matrix view.
   using MatrixView = Matrices::SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeRealType >;
   //! \brief Type of constant sparse matrix view.
   using ConstMatrixView = typename MatrixView::ConstViewType;

   //! \brief Base type.
   using Base = Matrices::SparseMatrixRowView< typename MatrixView::SegmentsViewType,
                                               typename MatrixView::ValuesViewType,
                                               typename MatrixView::ColumnIndexesViewType >;

   using ColumnIndexesViewType = typename Base::ColumnsIndexesViewType;  // TODO: Rename ColumnsIndexesViewType in Base

   using typename Base::IndexType;
   using typename Base::RealType;
   using typename Base::SegmentViewType;
   using typename Base::ValuesViewType;

   //! \brief Type of the graph node view.
   using NodeView = GraphNodeView< MatrixView, GraphType_ >;

   //! \brief Type of the constant graph node view.
   using ConstNodeView = GraphNodeView< ConstMatrixView, GraphType_ >;

   GraphNodeView( const SegmentViewType& segmentView,
                  const ValuesViewType& valuesView,
                  const ColumnIndexesViewType& columnIndexesView )
   : Base( segmentView, valuesView, columnIndexesView )
   {}

   [[nodiscard]] __cuda_callable__
   IndexType
   getNodeIndex() const
   {
      return Base::getRowIndex();
   }

   [[nodiscard]] __cuda_callable__
   const RealType&
   getWeight( IndexType edgeIndex ) const
   {
      return Base::getValue( edgeIndex );
   }

   [[nodiscard]] __cuda_callable__
   RealType&
   getWeight( IndexType edgeIndex )
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
   setWeight( IndexType edgeIndex, const RealType& weight )
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
};

}  // namespace TNL::Graphs
