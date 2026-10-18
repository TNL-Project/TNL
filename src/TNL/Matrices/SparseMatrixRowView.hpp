// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Matrices/SparseMatrixRowView.h>
#include <TNL/Assert.h>

namespace TNL::Matrices {

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::SparseMatrixRowView(
   SegmentViewType segmentView,
   ValuesViewType values,
   ColumnsIndexesViewType columnIndexes )
: segmentView( std::move( segmentView ) ),
  values( std::move( values ) ),
  columnIndexes( std::move( columnIndexes ) )
{}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::getSize() const -> IndexType
{
   return segmentView.getSize();
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::getRowIndex() const -> IndexType
{
   return segmentView.getSegmentIndex();
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::getColumnIndex( const IndexType localIdx ) const -> const
   typename ColumnsIndexesViewType::ValueType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::getColumnIndex( const IndexType localIdx ) ->
   typename ColumnsIndexesViewType::ValueType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::getValue( const IndexType localIdx ) const
   -> GetValueConstResultType
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );

   if constexpr( isBinary() ) {
      return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ] != paddingIndex< IndexType >;
   }
   else {
      return values[ segmentView.getGlobalIndex( localIdx ) ];
   }
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::getValue( const IndexType localIdx ) -> GetValueResultType
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );

   if constexpr( isBinary() ) {
      return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ] != paddingIndex< IndexType >;
   }
   else {
      return values[ segmentView.getGlobalIndex( localIdx ) ];
   }
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::setValue( const IndexType localIdx, const RealType& value )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   if constexpr( ! isBinary() ) {
      const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
      values[ globalIdx ] = value;
   }
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::setColumnIndex(
   const IndexType localIdx,
   const IndexType& columnIndex )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
   this->columnIndexes[ globalIdx ] = columnIndex;
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::setElement(
   const IndexType localIdx,
   const IndexType column,
   const RealType& value )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
   columnIndexes[ globalIdx ] = column;
   if constexpr( ! isBinary() )
      values[ globalIdx ] = value;
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::getGlobalIndex( IndexType localIdx ) const -> IndexType
{
   return segmentView.getGlobalIndex( localIdx );
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
template< typename _SegmentView, typename _ValuesView, typename _ColumnsIndexesView >
__cuda_callable__
bool
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::operator==(
   const SparseMatrixRowView< _SegmentView, _ValuesView, _ColumnsIndexesView >& other ) const
{
   // Only elements which are really present in the matrix are compared, i.e. the padding elements and the elements with
   // explicitly stored zero value are skipped. Two rows are therefore equal even if they differ in the explicitly stored
   // zero elements only. The elements are compared in the order in which they are stored in the rows, i.e. the rows do
   // not need to be sorted by the column indexes, but rows which store the same elements in a different order are not
   // equal.
   using OtherRowView = SparseMatrixRowView< _SegmentView, _ValuesView, _ColumnsIndexesView >;
   IndexType thisIdx = 0;
   IndexType otherIdx = 0;
   while( true ) {
      while( thisIdx < getSize() ) {
         if( getColumnIndex( thisIdx ) == paddingIndex< IndexType > ) {
            ++thisIdx;
            continue;
         }
         if constexpr( ! isBinary() ) {
            if( getValue( thisIdx ) == RealType{ 0 } ) {
               ++thisIdx;
               continue;
            }
         }
         break;
      }
      while( otherIdx < other.getSize() ) {
         if( other.getColumnIndex( otherIdx ) == paddingIndex< IndexType > ) {
            ++otherIdx;
            continue;
         }
         if constexpr( ! OtherRowView::isBinary() ) {
            if( other.getValue( otherIdx ) == typename OtherRowView::RealType{ 0 } ) {
               ++otherIdx;
               continue;
            }
         }
         break;
      }
      if( thisIdx == getSize() || otherIdx == other.getSize() )
         break;
      if( getColumnIndex( thisIdx ) != other.getColumnIndex( otherIdx ) )
         return false;
      if constexpr( ! isBinary() || ! OtherRowView::isBinary() ) {
         // A binary matrix stores only the values 1, so the non-binary side is compared with the value 1. This also
         // avoids mixing of the operands' types in the comparison, which would not compile e.g. for std::complex.
         if constexpr( isBinary() ) {
            if( other.getValue( otherIdx ) != typename OtherRowView::RealType{ 1 } )
               return false;
         }
         else if constexpr( OtherRowView::isBinary() ) {
            if( getValue( thisIdx ) != RealType{ 1 } )
               return false;
         }
         else {
            if( getValue( thisIdx ) != other.getValue( otherIdx ) )
               return false;
         }
      }
      ++thisIdx;
      ++otherIdx;
   }
   return thisIdx == getSize() && otherIdx == other.getSize();
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::sortColumnIndexes()
{
   // Sort the row by insertion sort
   IndexType size = this->getSize();
   for( IndexType i = 1; i < size; i++ ) {
      const IndexType columnIdx = getColumnIndex( i );
      if( columnIdx == paddingIndex< IndexType > )
         return;
      const RealType value = getValue( i );
      IndexType j = i;
      for( j = i; j > 0 && getColumnIndex( j - 1 ) > columnIdx; j-- ) {
         getColumnIndex( j ) = getColumnIndex( j - 1 );
         if constexpr( ! isBinary() )
            getValue( j ) = getValue( j - 1 );
      }
      getColumnIndex( j ) = columnIdx;
      if constexpr( ! isBinary() )
         getValue( j ) = value;
   }
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::begin() -> IteratorType
{
   return { *this, 0 };
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::end() -> IteratorType
{
   return { *this, this->getSize() };
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::cbegin() const -> ConstIteratorType
{
   return { *this, 0 };
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::cend() const -> ConstIteratorType
{
   return { *this, this->getSize() };
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
std::ostream&
operator<<( std::ostream& str, const SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >& row )
{
   using RowView = SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >;
   using NonConstIndex = std::remove_const_t< typename RowView::IndexType >;
   for( NonConstIndex i = 0; i < row.getSize(); i++ )
      if constexpr( RowView::isBinary() )
         // TODO: print only the column indices of non-zeros but not the values
         str << " [ " << row.getColumnIndex( i )
             << " ] = " << (row.getColumnIndex( i ) != paddingIndex< NonConstIndex >) << ", ";
      else
         str << " [ " << row.getColumnIndex( i ) << " ] = " << row.getValue( i ) << ", ";
   return str;
}

}  // namespace TNL::Matrices
