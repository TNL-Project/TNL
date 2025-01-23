// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "MatrixBase.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
__cuda_callable__
void
MatrixBase< Real, Device, Index, MatrixType, Organization >::bind( IndexType rows, IndexType columns, ValuesViewType values )
{
   this->rows = rows;
   this->columns = columns;
   this->values.bind( std::move( values ) );
}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
__cuda_callable__
MatrixBase< Real, Device, Index, MatrixType, Organization >::MatrixBase( IndexType rows,
                                                                         IndexType columns,
                                                                         ValuesViewType values )
: rows( rows ),
  columns( columns ),
  values( std::move( values ) )
{}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
Index
MatrixBase< Real, Device, Index, MatrixType, Organization >::getAllocatedElementsCount() const
{
   return this->values.getSize();
}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
Index
MatrixBase< Real, Device, Index, MatrixType, Organization >::getNonzeroElementsCount() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [ = ] __cuda_callable__( const IndexType i ) -> IndexType
   {
      return values_view[ i ] != RealType{ 0 };
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, this->values.getSize(), fetch, std::plus<>{}, 0 );
}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
__cuda_callable__
Index
MatrixBase< Real, Device, Index, MatrixType, Organization >::getRows() const
{
   return this->rows;
}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
__cuda_callable__
Index
MatrixBase< Real, Device, Index, MatrixType, Organization >::getColumns() const
{
   return this->columns;
}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
__cuda_callable__
const typename MatrixBase< Real, Device, Index, MatrixType, Organization >::ValuesViewType&
MatrixBase< Real, Device, Index, MatrixType, Organization >::getValues() const
{
   return this->values;
}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
__cuda_callable__
typename MatrixBase< Real, Device, Index, MatrixType, Organization >::ValuesViewType&
MatrixBase< Real, Device, Index, MatrixType, Organization >::getValues()
{
   return this->values;
}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
template< typename MatrixT >
bool
MatrixBase< Real, Device, Index, MatrixType, Organization >::operator==( const MatrixT& matrix ) const
{
   if( this->getRows() != matrix.getRows() || this->getColumns() != matrix.getColumns() )
      return false;
   for( IndexType row = 0; row < this->getRows(); row++ )
      for( IndexType column = 0; column < this->getColumns(); column++ )
         if( this->getElement( row, column ) != matrix.getElement( row, column ) )
            return false;
   return true;
}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
template< typename MatrixT >
bool
MatrixBase< Real, Device, Index, MatrixType, Organization >::operator!=( const MatrixT& matrix ) const
{
   return ! operator==( matrix );
}

}  // namespace TNL::Matrices
