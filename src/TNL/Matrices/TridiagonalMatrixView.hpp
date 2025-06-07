// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "TridiagonalMatrixView.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
TridiagonalMatrixView< Real, Device, Index, Organization >::TridiagonalMatrixView( typename Base::ValuesViewType values,
                                                                                   typename Base::IndexerType indexer )
: Base( std::move( values ), std::move( indexer ) )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
TridiagonalMatrixView< Real, Device, Index, Organization >::bind( TridiagonalMatrixView view )
{
   Base::bind( std::move( view.values ), std::move( view.indexer ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
auto
TridiagonalMatrixView< Real, Device, Index, Organization >::getView() -> ViewType
{
   return { this->getValues().getView(), this->getIndexer() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
auto
TridiagonalMatrixView< Real, Device, Index, Organization >::getConstView() const -> ConstViewType
{
   return { this->getValues().getConstView(), this->getIndexer() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
File&
operator>>( File& file, TridiagonalMatrixView< Real, Device, Index, Organization >& matrix )
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
operator>>( File&& file, TridiagonalMatrixView< Real, Device, Index, Organization >& matrix )
{
   // named r-value is an l-value reference, so this is not recursion
   return file >> matrix;
}

}  // namespace TNL::Matrices
