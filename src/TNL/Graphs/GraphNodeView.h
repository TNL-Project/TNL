// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseMatrixRowView.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseMatrixRowView.h>

namespace TNL::Graphs {

template< typename Matrix, typename GraphType_ >
struct Graph;

template< typename Graph >
struct GraphNodeView;

template< typename Real,
          typename Device,
          typename Index,
          Algorithms::Segments::ElementsOrganization Organization,
          typename RealAllocator,
          typename GraphType_ >
struct GraphNodeView< Graph< Matrices::DenseMatrix< Real, Device, Index, Organization, RealAllocator >, GraphType_ > >
: public Matrices::DenseMatrixRowView<
     typename Matrices::DenseMatrix< Real, Device, Index, Organization, RealAllocator >::SegmentsViewType,
     typename Matrices::DenseMatrix< Real, Device, Index, Organization, RealAllocator >::ValuesViewType >
{
   using MatrixType = Matrices::DenseMatrix< Real, Device, Index, Organization, RealAllocator >;

   using Base = Matrices::DenseMatrixRowView< typename MatrixType::SegmentsViewType, typename MatrixType::ValuesViewType >;

   using typename Base::IndexType;
   using typename Base::RealType;
   using typename Base::SegmentViewType;
   using typename Base::ValuesViewType;

   using GraphType = Graph< MatrixType, GraphType_ >;

   using NodeView = GraphNodeView< Graph< MatrixType, GraphType_ > >;

   using ConstNodeView = GraphNodeView< std::add_const_t< Graph< MatrixType, GraphType_ > > >;

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
          template< typename Device_, typename Index_, typename IndexAllocator_ > class SegmentsType,
          typename ComputeRealType,
          typename RealAllocator,
          typename IndexAllocator,
          typename GraphType_ >
struct GraphNodeView< Graph<
   Matrices::SparseMatrix< Real, Device, Index, MatrixType_, SegmentsType, ComputeRealType, RealAllocator, IndexAllocator >,
   GraphType_ > >
: public Matrices::SparseMatrixRowView<
     typename Matrices::
        SparseMatrix< Real, Device, Index, MatrixType_, SegmentsType, ComputeRealType, RealAllocator, IndexAllocator >::
           SegmentsViewType,
     typename Matrices::
        SparseMatrix< Real, Device, Index, MatrixType_, SegmentsType, ComputeRealType, RealAllocator, IndexAllocator >::
           ValuesViewType,
     typename Matrices::
        SparseMatrix< Real, Device, Index, MatrixType_, SegmentsType, ComputeRealType, RealAllocator, IndexAllocator >::
           ColumnIndexesViewType >
{
   using MatrixType =
      Matrices::SparseMatrix< Real, Device, Index, MatrixType_, SegmentsType, ComputeRealType, RealAllocator, IndexAllocator >;

   using Base = Matrices::SparseMatrixRowView< typename MatrixType::SegmentsViewType,
                                               typename MatrixType::ValuesViewType,
                                               typename MatrixType::ColumnIndexesViewType >;

   using ColumnIndexesViewType = typename Base::ColumnsIndexesViewType;  // TODO: Rename ColumnsIndexesViewType in Base
   using typename Base::IndexType;
   using typename Base::RealType;
   using typename Base::SegmentViewType;
   using typename Base::ValuesViewType;

   using GraphType = Graph< MatrixType, GraphType_ >;

   using NodeView = GraphNodeView< Graph< MatrixType, GraphType_ > >;

   using ConstNodeView = GraphNodeView< std::add_const_t< Graph< MatrixType, GraphType_ > > >;

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
