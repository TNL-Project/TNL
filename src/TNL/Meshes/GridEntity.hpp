// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridEntity.h>

namespace TNL {
namespace Meshes {

template< class Grid, int EntityDimension >
constexpr int
GridEntity< Grid, EntityDimension >::getMeshDimension()
{
   return Grid::getMeshDimension();
}

template< class Grid, int EntityDimension >
constexpr int
GridEntity< Grid, EntityDimension >::getEntityDimension()
{
   return EntityDimension;
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::GridEntity() : GridEntityBaseType( 0 ),  grid( nullptr )
{}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::GridEntity( const CoordinatesType& c ) : GridEntityBaseType( c ),  grid( nullptr )
{}

template< class Grid, int EntityDimension >
   template< typename Value >
__cuda_callable__
GridEntity< Grid, EntityDimension >::GridEntity( const std::initializer_list< Value >& elems )
   : GridEntityBaseType( elems ), grid( 0 )
{
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::GridEntity( const Grid& grid, const CoordinatesType& coordinates )
: GridEntityBaseType( coordinates ), grid( &grid )
{
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::GridEntity( const Grid& grid,
                                                 const CoordinatesType& coordinates,
                                                 const NormalsType& normals )
: GridEntityBaseType( coordinates, grid.getEntitiesOrientations().getTotalOrientationIndex( normals ) ),
  grid( &grid )
{
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::GridEntity( const Grid& grid, IndexType entityIdx ) : grid( &grid ), index( entityIdx )
{
   TNL_ASSERT_NE( this->grid, nullptr, "Grid pointer cannot be initialized with null pointer. Use setGrid method first." );
   IndexType totalOrientationIndex;
   auto coordinates = grid.template getEntityCoordinates< EntityDimension >( entityIdx, totalOrientationIndex );
   this->setTotalOrientationIndex( totalOrientationIndex );
   this->setCoordinates( coordinates );
   TNL_ASSERT_EQ( getNormals(), grid.getNormals( getTotalOrientationIndex() ), "Wrong index of entity orientation." );
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::GridEntity( const Grid& grid,
                                                 const CoordinatesType& coordinates,
                                                 const IndexType orientationIndex )
: GridEntityBaseType( coordinates, EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >( orientationIndex ) ),
  grid( &grid )
{
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
const typename GridEntity< Grid, EntityDimension >::CoordinatesType&
GridEntity< Grid, EntityDimension >::getCoordinates() const
{
   return GridEntityBaseType::getCoordinates();
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::CoordinatesType&
GridEntity< Grid, EntityDimension >::getCoordinates()
{
   return GridEntityBaseType::getCoordinates();
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setCoordinates( const CoordinatesType& coordinates )
{
   GridEntityBaseType::setCoordinates( coordinates );
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::refresh()
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   this->index = this->grid->getEntityIndex( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::getIndex() const -> const IndexType&
{
   TNL_ASSERT_GE( this->index, 0, "Entity index is not non-negative." );
   TNL_ASSERT_LT( this->index, grid->getEntitiesCount( EntityDimension ), "Entity index is out of bounds." );
   TNL_ASSERT_EQ( this->index, grid->getEntityIndex( *this ), "Wrong value of stored index." );

   return this->index;
}

template< class Grid, int EntityDimension >
__cuda_callable__
bool
GridEntity< Grid, EntityDimension >::isBoundary() const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return grid->isBoundaryEntity( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::PointType
GridEntity< Grid, EntityDimension >::getCenter() const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return grid->getEntityCenter( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::RealType
GridEntity< Grid, EntityDimension >::getMeasure() const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return grid->getEntityMeasure( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
const Grid&
GridEntity< Grid, EntityDimension >::getMesh() const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return *this->grid;
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::
getOrientationIndex() const -> IndexType
{
   return GridEntityBaseType::getOrientationIndex();
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::
setOrientationIndex( IndexType orientationIndex )
{
   TNL_ASSERT_LT( orientationIndex, ( EntitiesOrientations::template getOrientationsCount< EntityDimension >() ),
      "Wrong orientation index for grid entity with given dimension." );
   std::cerr << "EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >( orientationIndex ) = " <<
      EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >( orientationIndex ) << std::endl;
   GridEntityBaseType::setTotalOrientationIndex(
      EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >( orientationIndex ) );
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::
getTotalOrientationIndex() const -> IndexType
{
   return GridEntityBaseType::getTotalOrientationIndex();
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::
setTotalOrientationIndex( IndexType totalOrientationIndex )
{
   TNL_ASSERT_LT( totalOrientationIndex, EntitiesOrientations::getTotalOrientationsCount(),
      "Wrong total orientation index." );
   TNL_ASSERT_EQ( EntityDimension, EntitiesOrientations::getEntityDimension( totalOrientationIndex ),
      "Total orientation index does not agree with entity dimension." );
   GridEntityBaseType::setTotalOrientationIndex( totalOrientationIndex );
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::getNormals() const -> const NormalsType
{
   TNL_ASSERT_NE( this->grid, nullptr, "Attempt to dereference null pointer. Use setGrid method first." );
   return this->grid->getNormals( this->getTotalOrientationIndex() );
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::getBasis() const -> NormalsType
{
   return 1 - this->getNormals();
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >
GridEntity< Grid, EntityDimension >::
getEntity( const CoordinatesType& offset ) const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return this->grid->getEntity( *this, offset );
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::
getEntityIndex( const CoordinatesType& offset ) const -> IndexType
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return this->grid->getEntityIndex( *this, offset );
}

template< class Grid, int EntityDimension >
   template< int NeighbourEntityDimension >
__cuda_callable__
GridEntity< Grid, NeighbourEntityDimension >
GridEntity< Grid, EntityDimension >::
getEntity( const CoordinatesType& offset,
           const NormalsType& neighbourEntityOrientation ) const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return this->grid->template getEntity< NeighbourEntityDimension >( *this, offset, neighbourEntityOrientation );
}

template< class Grid, int EntityDimension >
   template< int NeighbourEntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::
getEntityIndex( const CoordinatesType& offset,
                IndexType neighbourEntityOrientation ) const -> IndexType
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return this->grid->template getEntityIndex< NeighbourEntityDimension >( *this, offset, neighbourEntityOrientation );
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::
getAdjacentCells( IndexType& closer, IndexType& remoter ) const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   this->grid->getAdjacentCells( *this, closer, remoter );
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::
getAdjacentFacesIndexes( CoordinatesType& closer, CoordinatesType& remoter ) const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   this->grid->getAdjacentFacesIndexes( *this, closer, remoter );
}

template< class Grid, int EntityDimension >
   template< int Direction, int Step >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::
getAdjacentEntityIndex() const -> IndexType
{
   return this->grid->template getAdjacentEntityIndex< Direction, Step >( *this );
}

template< class Grid, int EntityDimension >
auto
GridEntity< Grid, EntityDimension >::getPoint() const -> PointType
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return this->grid->getSpaceSteps() * this->getCoordinates();
}

template< class Grid, int EntityDimension >
__cuda_callable__
const Grid&
GridEntity< Grid, EntityDimension >::getGrid() const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer. Use setGrid method first." );
   return *this->grid;
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setGrid( const Grid& grid )
{
   this->grid = &grid;
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setMesh( const Grid& grid )
{
   this->grid = &grid;
}

template< class Grid, int EntityDimension >
std::ostream&
operator<<( std::ostream& str, const GridEntity< Grid, EntityDimension >& entity )
{
   str << "Entity dimension = " << EntityDimension << " coordinates = " << entity.getCoordinates()
       << " normals = " << entity.getNormals() << " index = " << entity.getIndex()
       << " orientationIndex = " << entity.getOrientationIndex();
   return str;
}

}  // namespace Meshes
}  // namespace TNL
