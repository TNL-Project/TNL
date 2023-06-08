// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <TNL/Matrices/MatrixView.h>

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index >
__cuda_callable__
MatrixView< Real, Device, Index >::MatrixView( Index rows, Index columns, typename Base::ValuesViewType values )
: Base( rows, columns, std::move( values ) )
{}

template< typename Real, typename Device, typename Index >
__cuda_callable__
void
MatrixView< Real, Device, Index >::bind( MatrixView& view )
{
   Base::bind( view.rows, view.columns, view.values );
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
void
MatrixView< Real, Device, Index >::bind( MatrixView&& view )
{
   Base::bind( view.rows, view.columns, view.values );
}

template< typename Real, typename Device, typename Index >
void
MatrixView< Real, Device, Index >::save( File& file ) const
{
   file.save( magic_number, strlen( magic_number ) );
   file << this->getSerializationTypeVirtual();
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->values;
}

template< typename Real, typename Device, typename Index >
void
MatrixView< Real, Device, Index >::save( const String& fileName ) const
{
   File file;
   file.open( fileName, std::ios_base::out );
   this->save( file );
}

template< typename Real, typename Device, typename Index >
void
MatrixView< Real, Device, Index >::print( std::ostream& str ) const
{}

}  // namespace TNL::Matrices
