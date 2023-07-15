// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "DenseMatrixView.h"

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
   Base::bind( view.getRows(), view.getColumns(), view.getValues() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
DenseMatrixView< Real, Device, Index, Organization >::bind( DenseMatrixView&& view )
{
   Base::bind( view.getRows(), view.getColumns(), view.getValues() );
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
void
DenseMatrixView< Real, Device, Index, Organization >::save( File& file ) const
{
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->getValues();
   this->segments.save( file );
}

}  // namespace TNL::Matrices
