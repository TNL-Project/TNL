// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>

namespace TNL::Containers {

template< typename Real, typename Device, typename Index, typename Allocator >
Vector< Real, Device, Index, Allocator >::Vector( const Vector& vector, const AllocatorType& allocator )
: Array< Real, Device, Index, Allocator >( vector, allocator )
{}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression, typename..., typename >
Vector< Real, Device, Index, Allocator >::Vector( const VectorExpression& expression )
{
   detail::VectorAssignment< Vector, VectorExpression >::resize( *this, expression );
   detail::VectorAssignment< Vector, VectorExpression >::assign( *this, expression );
}

template< typename Real, typename Device, typename Index, typename Allocator >
typename Vector< Real, Device, Index, Allocator >::ViewType
Vector< Real, Device, Index, Allocator >::getView( IndexType begin, IndexType end )
{
   if( end == 0 )
      end = this->getSize();

   if( begin < (Index) 0 || begin > end )
      throw std::out_of_range( "getView: begin is out of range" );
   if( end < (Index) 0 || end > this->getSize() )
      throw std::out_of_range( "getView: end is out of range" );

   return ViewType( this->getData() + begin, end - begin );
}

template< typename Real, typename Device, typename Index, typename Allocator >
typename Vector< Real, Device, Index, Allocator >::ConstViewType
Vector< Real, Device, Index, Allocator >::getConstView( IndexType begin, IndexType end ) const
{
   if( end == 0 )
      end = this->getSize();

   if( begin < (Index) 0 || begin > end )
      throw std::out_of_range( "getConstView: begin is out of range" );
   if( end < (Index) 0 || end > this->getSize() )
      throw std::out_of_range( "getConstView: end is out of range" );

   return ConstViewType( this->getData() + begin, end - begin );
}

template< typename Real, typename Device, typename Index, typename Allocator >
Vector< Real, Device, Index, Allocator >::
operator ViewType()
{
   return getView();
}

template< typename Real, typename Device, typename Index, typename Allocator >
Vector< Real, Device, Index, Allocator >::
operator ConstViewType() const
{
   return getConstView();
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression, typename..., typename >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator=( const VectorExpression& expression )
{
   detail::VectorAssignment< Vector, VectorExpression >::resize( *this, expression );
   detail::VectorAssignment< Vector, VectorExpression >::assign( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator+=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::addition( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator-=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::subtraction( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator*=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::multiplication( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator/=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::division( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator%=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::modulo( *this, expression );
   return *this;
}

}  // namespace TNL::Containers
