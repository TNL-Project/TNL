// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "DistributedVectorView.h"

namespace TNL::Containers {

template< typename Real, typename Device, typename Index >
typename DistributedVectorView< Real, Device, Index >::LocalViewType
DistributedVectorView< Real, Device, Index >::getLocalView()
{
   return BaseType::getLocalView();
}

template< typename Real, typename Device, typename Index >
typename DistributedVectorView< Real, Device, Index >::ConstLocalViewType
DistributedVectorView< Real, Device, Index >::getConstLocalView() const
{
   return BaseType::getConstLocalView();
}

template< typename Real, typename Device, typename Index >
typename DistributedVectorView< Real, Device, Index >::LocalViewType
DistributedVectorView< Real, Device, Index >::getLocalViewWithGhosts()
{
   return BaseType::getLocalViewWithGhosts();
}

template< typename Real, typename Device, typename Index >
typename DistributedVectorView< Real, Device, Index >::ConstLocalViewType
DistributedVectorView< Real, Device, Index >::getConstLocalViewWithGhosts() const
{
   return BaseType::getConstLocalViewWithGhosts();
}

template< typename Value, typename Device, typename Index >
typename DistributedVectorView< Value, Device, Index >::ViewType
DistributedVectorView< Value, Device, Index >::getView()
{
   return *this;
}

template< typename Value, typename Device, typename Index >
typename DistributedVectorView< Value, Device, Index >::ConstViewType
DistributedVectorView< Value, Device, Index >::getConstView() const
{
   return *this;
}

/*
 * Usual Vector methods follow below.
 */

template< typename Real, typename Device, typename Index >
template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator=( const Vector& vector )
{
   if( this->getSize() != vector.getSize() )
      throw std::logic_error( "operator=: the sizes of the array views must be equal, views are not resizable." );
   if( this->getLocalRange() != vector.getLocalRange() )
      throw std::logic_error( "operator=: the local ranges must be equal, views are not resizable." );
   if( this->getGhosts() != vector.getGhosts() )
      throw std::logic_error( "operator=: ghosts must be equal, views are not resizable." );
   if( this->getCommunicator() != vector.getCommunicator() )
      throw std::logic_error( "operator=: the communicators of the array views must be equal." );

   if( this->getCommunicator() != MPI_COMM_NULL ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() = vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator+=( const Vector& vector )
{
   if( this->getSize() != vector.getSize() )
      throw std::logic_error( "operator+=: the sizes of the array views must be equal, views are not resizable." );
   if( this->getLocalRange() != vector.getLocalRange() )
      throw std::logic_error( "operator+=: the local ranges must be equal, views are not resizable." );
   if( this->getGhosts() != vector.getGhosts() )
      throw std::logic_error( "operator+=: ghosts must be equal, views are not resizable." );
   if( this->getCommunicator() != vector.getCommunicator() )
      throw std::logic_error( "operator+=: the communicators of the array views must be equal." );

   if( this->getCommunicator() != MPI_COMM_NULL ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() += vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator-=( const Vector& vector )
{
   if( this->getSize() != vector.getSize() )
      throw std::logic_error( "operator-=: the sizes of the array views must be equal, views are not resizable." );
   if( this->getLocalRange() != vector.getLocalRange() )
      throw std::logic_error( "operator-=: the local ranges must be equal, views are not resizable." );
   if( this->getGhosts() != vector.getGhosts() )
      throw std::logic_error( "operator-=: ghosts must be equal, views are not resizable." );
   if( this->getCommunicator() != vector.getCommunicator() )
      throw std::logic_error( "operator-=: the communicators of the array views must be equal." );

   if( this->getCommunicator() != MPI_COMM_NULL ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() -= vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator*=( const Vector& vector )
{
   if( this->getSize() != vector.getSize() )
      throw std::logic_error( "operator*=: the sizes of the array views must be equal, views are not resizable." );
   if( this->getLocalRange() != vector.getLocalRange() )
      throw std::logic_error( "operator*=: the local ranges must be equal, views are not resizable." );
   if( this->getGhosts() != vector.getGhosts() )
      throw std::logic_error( "operator*=: ghosts must be equal, views are not resizable." );
   if( this->getCommunicator() != vector.getCommunicator() )
      throw std::logic_error( "operator*=: the communicators of the array views must be equal." );

   if( this->getCommunicator() != MPI_COMM_NULL ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() *= vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator/=( const Vector& vector )
{
   if( this->getSize() != vector.getSize() )
      throw std::logic_error( "operator/=: the sizes of the array views must be equal, views are not resizable." );
   if( this->getLocalRange() != vector.getLocalRange() )
      throw std::logic_error( "operator/=: the local ranges must be equal, views are not resizable." );
   if( this->getGhosts() != vector.getGhosts() )
      throw std::logic_error( "operator/=: ghosts must be equal, views are not resizable." );
   if( this->getCommunicator() != vector.getCommunicator() )
      throw std::logic_error( "operator/=: the communicators of the array views must be equal." );

   if( this->getCommunicator() != MPI_COMM_NULL ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() /= vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Vector, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator%=( const Vector& vector )
{
   if( this->getSize() != vector.getSize() )
      throw std::logic_error( "operator%=: the sizes of the array views must be equal, views are not resizable." );
   if( this->getLocalRange() != vector.getLocalRange() )
      throw std::logic_error( "operator%=: the local ranges must be equal, views are not resizable." );
   if( this->getGhosts() != vector.getGhosts() )
      throw std::logic_error( "operator%=: ghosts must be equal, views are not resizable." );
   if( this->getCommunicator() != vector.getCommunicator() )
      throw std::logic_error( "operator%=: the communicators of the array views must be equal." );

   if( this->getCommunicator() != MPI_COMM_NULL ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      vector.waitForSynchronization();
      getLocalViewWithGhosts() %= vector.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator=( Scalar c )
{
   if( this->getCommunicator() != MPI_COMM_NULL ) {
      getLocalView() = c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator+=( Scalar c )
{
   if( this->getCommunicator() != MPI_COMM_NULL ) {
      getLocalView() += c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator-=( Scalar c )
{
   if( this->getCommunicator() != MPI_COMM_NULL ) {
      getLocalView() -= c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator*=( Scalar c )
{
   if( this->getCommunicator() != MPI_COMM_NULL ) {
      getLocalView() *= c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator/=( Scalar c )
{
   if( this->getCommunicator() != MPI_COMM_NULL ) {
      getLocalView() /= c;
      this->startSynchronization();
   }
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename Scalar, typename..., typename >
DistributedVectorView< Real, Device, Index >&
DistributedVectorView< Real, Device, Index >::operator%=( Scalar c )
{
   if( this->getCommunicator() != MPI_COMM_NULL ) {
      getLocalView() %= c;
      this->startSynchronization();
   }
   return *this;
}

}  // namespace TNL::Containers
