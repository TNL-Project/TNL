// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>

#include <TNL/MPI/Wrappers.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>

namespace TNL::Meshes::DistributedMeshes {

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::setDomainDecomposition( const CoordinatesType& domainDecomposition )
{
   this->domainDecomposition = domainDecomposition;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getDomainDecomposition() const
{
   return this->domainDecomposition;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::setGlobalGrid( const GridType& globalGrid )
{
   // use MPI only if have more than one process
   const bool distributed = MPI::GetSize( communicator ) > 1;

   // compute the decomposition and check that every subdomain gets at least one cell
   CoordinatesType decomposition = 1;
   if( distributed ) {
      int dims[ Dimension ];
      for( int i = 0; i < Dimension; i++ )
         dims[ i ] = this->domainDecomposition[ i ];
      MPI::Compute_dims( MPI::GetSize( communicator ), Dimension, dims );
      for( int i = 0; i < Dimension; i++ )
         decomposition[ i ] = dims[ i ];
   }
   for( int i = 0; i < Dimension; i++ )
      if( globalGrid.getSizes()[ i ] < decomposition[ i ] )
         throw std::invalid_argument(
            "DistributedGrid: the global grid has less cells than subdomains along axis " + std::to_string( i ) + "." );

   this->globalGrid = globalGrid;
   this->isSet = true;
   this->distributed = distributed;
   this->domainDecomposition = decomposition;

   if( ! this->distributed ) {
      this->subdomainCoordinates.setValue( 0 );
      this->localSize = globalGrid.getSizes();
      this->globalBegin = 0;
   }
   else {
      // the subdomain coordinates, the x-coordinate changes the fastest with the rank
      int size = MPI::GetSize( communicator );
      int tmp = MPI::GetRank( communicator );
      for( int i = Dimension - 1; i >= 0; i-- ) {
         size = size / this->domainDecomposition[ i ];
         this->subdomainCoordinates[ i ] = tmp / size;
         tmp = tmp % size;
      }

      // the first subdomains along each axis get one cell more if the sizes are not divisible
      for( int i = 0; i < Dimension; i++ ) {
         const Index numberOfLarger = globalGrid.getSizes()[ i ] % this->domainDecomposition[ i ];
         this->localSize[ i ] = globalGrid.getSizes()[ i ] / this->domainDecomposition[ i ];
         if( numberOfLarger > this->subdomainCoordinates[ i ] ) {
            this->localSize[ i ] += 1;
            this->globalBegin[ i ] = this->subdomainCoordinates[ i ] * this->localSize[ i ];
         }
         else
            this->globalBegin[ i ] = numberOfLarger * ( this->localSize[ i ] + 1 )
                                   + ( this->subdomainCoordinates[ i ] - numberOfLarger ) * this->localSize[ i ];
      }
   }
   this->setupNeighbors();
   this->setOverlaps( 0, 0 );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::setOverlaps(
   const SubdomainOverlapsType& lower,
   const SubdomainOverlapsType& upper )
{
   if( ! this->isSet )
      throw std::logic_error( "DistributedGrid: the global grid must be set before the overlaps." );
   typename GridType::BoundarySidesType lowerGlobalBoundarySides;
   typename GridType::BoundarySidesType upperGlobalBoundarySides;
   for( int i = 0; i < Dimension; i++ ) {
      CoordinatesType direction = 0;
      direction[ i ] = -1;
      lowerGlobalBoundarySides[ i ] = ! this->isThereNeighbor( direction );
      direction[ i ] = 1;
      upperGlobalBoundarySides[ i ] = ! this->isThereNeighbor( direction );
      if( ( lowerGlobalBoundarySides[ i ] && lower[ i ] != 0 ) || ( upperGlobalBoundarySides[ i ] && upper[ i ] != 0 ) )
         throw std::invalid_argument( "DistributedGrid: overlaps on the global boundary are not supported." );
      if( lower[ i ] < 0 || upper[ i ] < 0 )
         throw std::invalid_argument( "DistributedGrid: overlaps must be non-negative." );
      // The ghost entities must be owned by the neighbors, i.e. the overlaps must not reach the entities owned by
      // the neighbors of the neighbors. All subdomains along the axis have at least minimalSize cells.
      const Index minimalSize = this->globalGrid.getSizes()[ i ] / this->domainDecomposition[ i ];
      if( ( lower[ i ] > 0 && lower[ i ] >= minimalSize ) || ( upper[ i ] > 0 && upper[ i ] >= minimalSize ) )
         throw std::invalid_argument( "DistributedGrid: overlaps must be smaller than the sizes of all subdomains." );
   }

   this->lowerOverlap = lower;
   this->upperOverlap = upper;
   localGrid.setSizes( this->localSize + this->lowerOverlap + this->upperOverlap );
   localGrid.setOrigin(
      this->globalGrid.getOrigin() + this->globalGrid.getSpaceSteps() * ( this->globalBegin - this->lowerOverlap ) );
   // setting space steps computes the grid proportions as a side effect
   localGrid.setSpaceSteps( globalGrid.getSpaceSteps() );

   // the owned cells and the sides on the global boundary
   localGrid.setLocalBegin( this->lowerOverlap );
   localGrid.setLocalEnd( localGrid.getSizes() - this->upperOverlap );
   localGrid.setGlobalBoundarySides( lowerGlobalBoundarySides, upperGlobalBoundarySides );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::setGhostLevels( int levels )
{
   SubdomainOverlapsType lower = 0;
   SubdomainOverlapsType upper = 0;
   for( int i = 0; i < Dimension; i++ ) {
      CoordinatesType direction = 0;
      direction[ i ] = -1;
      if( this->isThereNeighbor( direction ) )
         lower[ i ] = levels;
      direction[ i ] = 1;
      if( this->isThereNeighbor( direction ) )
         upper[ i ] = levels;
   }
   setOverlaps( lower, upper );
}

template< int Dimension, typename Real, typename Device, typename Index >
int
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getGhostLevels() const
{
   return TNL::max( TNL::max( lowerOverlap ), TNL::max( upperOverlap ) );
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getSubdomainCoordinates() const
{
   return this->subdomainCoordinates;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::isDistributed() const
{
   return this->distributed;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::isBoundarySubdomain() const
{
   for( int i = 0; i < getNeighborsCount(); i++ )
      if( this->neighbors[ i ] == -1 )
         return true;
   return false;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getLowerOverlap() const
{
   return this->lowerOverlap;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getUpperOverlap() const
{
   return this->upperOverlap;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::GridType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getLocalMesh() const
{
   return this->localGrid;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getLocalSize() const
{
   return this->localSize;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getGlobalSize() const
{
   return this->globalGrid.getSizes();
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::GridType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getGlobalGrid() const
{
   return this->globalGrid;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getGlobalBegin() const
{
   return this->globalBegin;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
Index
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getEntitiesCount() const
{
   return this->globalGrid.template getEntitiesCount< EntityDimension >();
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
Index
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getEntitiesCount() const
{
   return this->globalGrid.template getEntitiesCount< Entity >();
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::setCommunicator( const MPI::Comm& communicator )
{
   this->communicator = communicator;
}

template< int Dimension, typename Real, typename Device, typename Index >
const MPI::Comm&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getCommunicator() const
{
   return this->communicator;
}

template< int Dimension, typename Real, typename Device, typename Index >
int
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getRankOfProcCoord( const CoordinatesType& nodeCoordinates ) const
{
   int dimensionOffset = 1;
   int ret = 0;
   for( int i = 0; i < Dimension; i++ ) {
      ret += dimensionOffset * nodeCoordinates[ i ];
      dimensionOffset *= this->domainDecomposition[ i ];
   }
   return ret;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::isThereNeighbor( const CoordinatesType& direction ) const
{
   bool res = true;
   for( int i = 0; i < Dimension; i++ ) {
      if( direction[ i ] == -1 )
         res &= this->subdomainCoordinates[ i ] > 0;
      if( direction[ i ] == 1 )
         res &= this->subdomainCoordinates[ i ] < this->domainDecomposition[ i ] - 1;
   }
   return res;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::setupNeighbors()
{
   for( int i = 0; i < getNeighborsCount(); i++ ) {
      const CoordinatesType direction( Directions::template getXYZ< Dimension >( i ) );
      if( this->isThereNeighbor( direction ) )
         this->neighbors[ i ] = this->getRankOfProcCoord( this->subdomainCoordinates + direction );
      else
         this->neighbors[ i ] = -1;
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::NeighborsType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::getNeighbors() const
{
   TNL_ASSERT_TRUE( this->isSet, "DistributedGrid is not set, but used by getNeighbors" );
   return this->neighbors;
}

template< int Dimension, typename Real, typename Device, typename Index >
String
DistributedMesh< Grid< Dimension, Real, Device, Index > >::printProcessCoords() const
{
   String res = convertToString( this->subdomainCoordinates[ 0 ] );
   for( int i = 1; i < Dimension; i++ )
      res = res + String( "-" ) + convertToString( this->subdomainCoordinates[ i ] );
   return res;
}

template< int Dimension, typename Real, typename Device, typename Index >
String
DistributedMesh< Grid< Dimension, Real, Device, Index > >::printProcessDistr() const
{
   String res = convertToString( this->domainDecomposition[ 0 ] );
   for( int i = 1; i < Dimension; i++ )
      res = res + String( "-" ) + convertToString( this->domainDecomposition[ i ] );
   return res;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::writeProlog( Logger& logger )
{
   logger.writeParameter( "Domain decomposition:", this->getDomainDecomposition() );
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::operator==( const DistributedMesh& other ) const
{
   return globalGrid == other.globalGrid && localGrid == other.localGrid && localSize == other.localSize
       && globalBegin == other.globalBegin && lowerOverlap == other.lowerOverlap && upperOverlap == other.upperOverlap
       && domainDecomposition == other.domainDecomposition && subdomainCoordinates == other.subdomainCoordinates
       && distributed == other.distributed && isSet == other.isSet && communicator == other.communicator;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::operator!=( const DistributedMesh& other ) const
{
   return ! operator==( other );
}

template< int Dimension, typename Real, typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str, const DistributedMesh< Grid< Dimension, Real, Device, Index > >& grid )
{
   const MPI::Comm& communicator = grid.getCommunicator();
   for( int j = 0; j < communicator.size(); j++ ) {
      if( j == communicator.rank() ) {
         str << "Node : " << communicator.rank() << '\n'
             << " globalGrid : " << grid.getGlobalGrid() << '\n'
             << " localGrid : " << grid.getLocalMesh() << '\n'
             << " localSize : " << grid.getLocalSize() << '\n'
             << " globalBegin : " << grid.getGlobalBegin() << '\n'
             << " lowerOverlap : " << grid.getLowerOverlap() << '\n'
             << " upperOverlap : " << grid.getUpperOverlap() << '\n'
             << " domainDecomposition : " << grid.getDomainDecomposition() << '\n'
             << " subdomainCoordinates : " << grid.getSubdomainCoordinates() << '\n'
             << " neighbors : ";
         for( int i = 0; i < grid.getNeighborsCount(); i++ )
            str << " " << grid.getNeighbors()[ i ];
         str << '\n';
      }
      communicator.barrier();
   }
   return str;
}

}  // namespace TNL::Meshes::DistributedMeshes
