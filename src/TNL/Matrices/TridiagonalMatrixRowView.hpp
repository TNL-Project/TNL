// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/TridiagonalMatrixRowView.h>

namespace TNL::Matrices {

template< typename ValuesView, typename Indexer >
__cuda_callable__
TridiagonalMatrixRowView< ValuesView, Indexer >::TridiagonalMatrixRowView( IndexType rowIdx,
                                                                           ValuesViewType values,
                                                                           IndexerType indexer )
: rowIdx( rowIdx ),
  values( std::move( values ) ),
  indexer( std::move( indexer ) )
{}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::getSize() const -> IndexType
{
   return indexer.getRowSize( rowIdx );
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::getRowIndex() const -> IndexType
{
   return rowIdx;
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::getColumnIndex( const IndexType localIdx ) const -> IndexType
{
   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, 3, "" );
   return rowIdx + localIdx - 1;
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::getValue( const IndexType localIdx ) const -> const RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::getValue( const IndexType localIdx ) -> RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
void
TridiagonalMatrixRowView< ValuesView, Indexer >::setElement( const IndexType localIdx, const RealType& value )
{
   this->values[ indexer.getGlobalIndex( rowIdx, localIdx ) ] = value;
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::begin() -> IteratorType
{
   return { *this, 0 };
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::end() -> IteratorType
{
   return { *this, this->getSize() };
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::cbegin() const -> ConstIteratorType
{
   return { *this, 0 };
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::cend() const -> ConstIteratorType
{
   return { *this, this->getSize() };
}

}  // namespace TNL::Matrices
