// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "DenseMatrixView.h"
#include "DenseOperations.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
DenseMatrixView< Real, Device, Index, Organization >::DenseMatrixView( Index rows,
                                                                       Index columns,
                                                                       typename Base::ValuesViewType values )
: Base( rows, columns, std::move( values ) )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
DenseMatrixView< Real, Device, Index, Organization >::bind( DenseMatrixView& view )
{
   Base::bind( view.getRows(), view.getColumns(), view.getValues(), view.segments );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
DenseMatrixView< Real, Device, Index, Organization >::bind( DenseMatrixView&& view )
{
   Base::bind( view.getRows(), view.getColumns(), view.getValues(), view.segments );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::getView() -> ViewType
{
   return ViewType( this->getRows(), this->getColumns(), this->getValues().getView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::getConstView() const -> ConstViewType
{
   return ConstViewType( this->getRows(), this->getColumns(), this->getValues().getConstView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< int tileDim >
void
DenseMatrixView< Real, Device, Index, Organization >::getInPlaceTransposition( Real matrixMultiplicator )
{
   TNL::Matrices::getInPlaceTransposition( *this, matrixMultiplicator );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
File&
operator>>( File& file, DenseMatrixView< Real, Device, Index, Organization >& matrix )
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
   file >> matrix.getValues();
   return file;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
File&
operator>>( File&& file, DenseMatrixView< Real, Device, Index, Organization >& matrix )
{
   // named r-value is an l-value reference, so this is not recursion
   return file >> matrix;
}

}  // namespace TNL::Matrices
