// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SparseMatrixView.h"

namespace TNL::Matrices {

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType_,
          template< typename, typename > class SegmentsView,
          typename ComputeReal >
__cuda_callable__
SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >::SparseMatrixView(
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
          typename MatrixType_,
          template< typename, typename > class SegmentsView,
          typename ComputeReal >
__cuda_callable__
void
SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >::bind( SparseMatrixView& view )
{
   Base::bind( view.getRows(), view.getColumns(), view.getValues(), view.getColumnIndexes(), view.getSegments() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType_,
          template< typename, typename > class SegmentsView,
          typename ComputeReal >
__cuda_callable__
void
SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >::bind( SparseMatrixView&& view )
{
   Base::bind( view.getRows(), view.getColumns(), view.getValues(), view.getColumnIndexes(), view.getSegments() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType_,
          template< typename, typename > class SegmentsView,
          typename ComputeReal >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >::getView() -> ViewType
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
          typename MatrixType_,
          template< typename, typename > class SegmentsView,
          typename ComputeReal >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >::getConstView() const -> ConstViewType
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
          typename MatrixType_,
          template< typename, typename > class SegmentsView,
          typename ComputeReal >
File&
operator>>( File& file, SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >& matrix )
{
   const std::string type = getObjectType( file );
   if( type != matrix.getSerializationType() )
      throw Exceptions::FileDeserializationError( file.getFileName(),
                                                  "object type does not match (expected " + matrix.getSerializationType()
                                                     + ", found " + type + ")." );
   std::size_t rows = 0;
   std::size_t columns = 0;
   file.load( &rows );
   file.load( &columns );
   if( rows != static_cast< std::size_t >( matrix.getRows() ) )
      throw Exceptions::FileDeserializationError( file.getFileName(),
                                                  "invalid number of rows: " + std::to_string( rows ) + " (expected "
                                                     + std::to_string( matrix.getRows() ) + ")." );
   if( columns != static_cast< std::size_t >( matrix.getColumns() ) )
      throw Exceptions::FileDeserializationError( file.getFileName(),
                                                  "invalid number of columns: " + std::to_string( columns ) + " (expected "
                                                     + std::to_string( matrix.getColumns() ) + ")." );
   matrix.getSegments().load( file );
   file >> matrix.getValues() >> matrix.getColumnIndexes();
   return file;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType_,
          template< typename, typename > class SegmentsView,
          typename ComputeReal >
File&
operator>>( File&& file, SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsView, ComputeReal >& matrix )
{
   // named r-value is an l-value reference, so this is not recursion
   return file >> matrix;
}

}  // namespace TNL::Matrices
