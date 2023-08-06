// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Matrices/SparseMatrixRowView.h>
#include <TNL/Assert.h>

namespace TNL::Matrices {

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::SparseMatrixRowView(
   const SegmentViewType& segmentView,
   const ValuesViewType& values,
   const ColumnsIndexesViewType& columnIndexes )
: segmentView( segmentView ), values( values ), columnIndexes( columnIndexes )
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
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::setColumnIndex( const IndexType localIdx,
                                                                                    const IndexType& columnIndex )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
   this->columnIndexes[ globalIdx ] = columnIndex;
}

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView >
__cuda_callable__
void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::setElement( const IndexType localIdx,
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
   IndexType i = 0;
   while( i < getSize() && i < other.getSize() ) {
      if( getColumnIndex( i ) != other.getColumnIndex( i ) )
         return false;
      if( ! isBinary() && getValue( i ) != other.getValue( i ) )
         return false;
      ++i;
   }
   for( IndexType j = i; j < getSize(); j++ )
      if( getColumnIndex( j ) != paddingIndex< IndexType > )
         return false;
   for( IndexType j = i; j < other.getSize(); j++ )
      if( other.getColumnIndex( j ) != paddingIndex< IndexType > )
         return false;
   return true;
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
         if( ! isBinary() )
            getValue( j ) = getValue( j - 1 );
      }
      getColumnIndex( j ) = columnIdx;
      if( ! isBinary() )
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
   using NonConstIndex =
      std::remove_const_t< typename SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::IndexType >;
   for( NonConstIndex i = 0; i < row.getSize(); i++ )
      if constexpr( row.isBinary() )
         // TODO: print only the column indices of non-zeros but not the values
         str << " [ " << row.getColumnIndex( i )
             << " ] = " << (row.getColumnIndex( i ) != paddingIndex< NonConstIndex >) << ", ";
      else
         str << " [ " << row.getColumnIndex( i ) << " ] = " << row.getValue( i ) << ", ";
   return str;
}

}  // namespace TNL::Matrices
