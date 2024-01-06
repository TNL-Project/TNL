// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SparseMatrixView.h"

namespace TNL::Matrices {

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::SparseMatrixView(
   Index rows,
   Index columns,
   typename Base::ValuesViewType values,
   typename Base::ColumnIndexesViewType columnIndexes,
   typename Base::SegmentsViewType segments )
: Base( rows, columns, std::move( values ), std::move( columnIndexes ), std::move( segments ) )
{}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::bind( SparseMatrixView& view )
{
   Base::bind( view.getRows(), view.getColumns(), view.getValues(), view.getColumnIndexes(), view.getSegments() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::bind( SparseMatrixView&& view )
{
   Base::bind( view.getRows(), view.getColumns(), view.getValues(), view.getColumnIndexes(), view.getSegments() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getView() -> ViewType
{
   return { this->getRows(),
            this->getColumns(),
            this->getValues().getView(),
            this->getColumnIndexes().getView(),
            this->getSegments().getView() };
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getConstView() const -> ConstViewType
{
   return { this->getRows(),
            this->getColumns(),
            this->getValues().getConstView(),
            this->getColumnIndexes().getConstView(),
            this->getSegments().getConstView() };
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::save( File& file ) const
{
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->getValues() << this->getColumnIndexes();
   this->getSegments().save( file );
}

}  // namespace TNL::Matrices
