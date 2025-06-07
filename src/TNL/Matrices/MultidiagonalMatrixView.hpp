// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "MultidiagonalMatrixView.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
MultidiagonalMatrixView< Real, Device, Index, Organization >::MultidiagonalMatrixView(
   typename Base::ValuesViewType values,
   typename Base::DiagonalOffsetsView diagonalOffsets,
   typename Base::HostDiagonalOffsetsView hostDiagonalOffsets,
   typename Base::IndexerType indexer )
: Base( std::move( values ), std::move( diagonalOffsets ), std::move( hostDiagonalOffsets ), std::move( indexer ) )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::bind( MultidiagonalMatrixView view )
{
   Base::bind( std::move( view.values ),
               std::move( view.diagonalOffsets ),
               std::move( view.hostDiagonalOffsets ),
               std::move( view.indexer ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::getView() -> ViewType
{
   return {
      this->getValues().getView(), this->diagonalOffsets.getView(), this->hostDiagonalOffsets.getView(), this->getIndexer()
   };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::getConstView() const -> ConstViewType
{
   return { this->getValues.getConstView(),
            this->diagonalOffsets.getConstView(),
            this->hostDiagonalOffsets.getConstView(),
            this->getIndexer() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
File&
operator>>( File& file, MultidiagonalMatrixView< Real, Device, Index, Organization >& matrix )
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
   file >> matrix.getDiagonalOffsets() >> matrix.getValues();
   return file;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
File&
operator>>( File&& file, MultidiagonalMatrixView< Real, Device, Index, Organization >& matrix )
{
   // named r-value is an l-value reference, so this is not recursion
   return file >> matrix;
}

}  // namespace TNL::Matrices
