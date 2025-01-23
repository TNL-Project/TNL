// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/LambdaMatrixRowView.h>
#include <TNL/Assert.h>

namespace TNL::Matrices {

template< typename RowView >
__cuda_callable__
LambdaMatrixRowViewIterator< RowView >::LambdaMatrixRowViewIterator( const RowViewType& rowView, const IndexType& localIdx )
: rowView( rowView ),
  localIdx( localIdx )
{}

template< typename RowView >
__cuda_callable__
bool
LambdaMatrixRowViewIterator< RowView >::operator==( const LambdaMatrixRowViewIterator& other ) const
{
   return &this->rowView == &other.rowView && localIdx == other.localIdx;
}

template< typename RowView >
__cuda_callable__
bool
LambdaMatrixRowViewIterator< RowView >::operator!=( const LambdaMatrixRowViewIterator& other ) const
{
   return ! ( other == *this );
}

template< typename RowView >
__cuda_callable__
LambdaMatrixRowViewIterator< RowView >&
LambdaMatrixRowViewIterator< RowView >::operator++()
{
   if( localIdx < rowView.getSize() )
      localIdx++;
   return *this;
}

template< typename RowView >
__cuda_callable__
LambdaMatrixRowViewIterator< RowView >&
LambdaMatrixRowViewIterator< RowView >::operator--()
{
   if( localIdx > 0 )
      localIdx--;
   return *this;
}

template< typename RowView >
__cuda_callable__
auto
LambdaMatrixRowViewIterator< RowView >::operator*() -> MatrixElementType
{
   return MatrixElementType( this->rowView.getValue( this->localIdx ),
                             this->rowView.getRowIndex(),
                             this->rowView.getColumnIndex( this->localIdx ),
                             this->localIdx );
}

template< typename RowView >
__cuda_callable__
auto
LambdaMatrixRowViewIterator< RowView >::operator*() const -> MatrixElementType
{
   return MatrixElementType( this->rowView.getValue( this->localIdx ),
                             this->rowView.getRowIndex(),
                             this->rowView.getColumnIndex( this->localIdx ),
                             this->localIdx );
}

}  // namespace TNL::Matrices
