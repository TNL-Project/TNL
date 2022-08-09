// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridDetails/BoundaryGridEntityChecker.h>
#include <TNL/Meshes/GridDetails/GridEntityCenterGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter.h>

namespace TNL {
namespace Meshes {

template< class Grid, int EntityDimension >
constexpr int
GridEntity< Grid, EntityDimension >::
getMeshDimension()
{
   return meshDimension;
}

template< class Grid, int EntityDimension >
constexpr int
GridEntity< Grid, EntityDimension >::
getEntityDimension()
{
   return entityDimension;
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::
GridEntity( const Grid& grid )
: grid( grid ), coordinates( 0 )
{
   this->normals = grid.template getNormals<EntityDimension>(0);
   this->orientation = 0;
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::
GridEntity( const Grid& grid, const CoordinatesType& coordinates )
: grid( grid ), coordinates( coordinates )
{
   normals = grid.template getNormals<EntityDimension>(0);
   orientation = 0;
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::
GridEntity( const Grid& grid, const CoordinatesType& coordinates, const CoordinatesType& normals )
: grid( grid ), coordinates( coordinates ), normals( normals ), 
   orientation( grid.template getOrientation< EntityDimension >( normals ) )
{
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::
GridEntity( const Grid& grid, const CoordinatesType& coordinates, const CoordinatesType& normals, 
   const IndexType orientation )
: grid( grid ), coordinates( coordinates ), normals( normals ), orientation( orientation )
{
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
const typename GridEntity< Grid, EntityDimension >::CoordinatesType&
GridEntity< Grid, EntityDimension >::getCoordinates() const
{
   return this->coordinates;
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::CoordinatesType&
GridEntity< Grid, EntityDimension >::getCoordinates()
{
   return this->coordinates;
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::refresh()
{
   this->index = this->grid.getEntityIndex( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::IndexType
GridEntity< Grid, EntityDimension >::getIndex() const
{
   TNL_ASSERT_GE( this->index, 0, "Entity index is not non-negative." );
   TNL_ASSERT_LT( this->index, grid.template getEntitiesCount< EntityDimension >(), "Entity index is out of bounds." );
   TNL_ASSERT_EQ( this->index, grid.getEntityIndex( *this ), "Wrong value of stored index." );

   return this->index;
}

template< class Grid, int EntityDimension >
__cuda_callable__
bool
GridEntity< Grid, EntityDimension >::isBoundary() const
{
   return BoundaryGridEntityChecker< GridEntity >::isBoundaryEntity( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
const typename GridEntity< Grid, EntityDimension >::PointType
GridEntity< Grid, EntityDimension >::getCenter() const
{
   return GridEntityCenterGetter< GridEntity >::getEntityCenter( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::RealType
GridEntity< Grid, EntityDimension >::getMeasure() const
{
   return GridEntityMeasureGetter< Grid, EntityDimension >::getMeasure( this->getMesh(), *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
const Grid&
GridEntity< Grid, EntityDimension >::getMesh() const
{
   return this->grid;
}

template< class Grid, int EntityDimension >
__cuda_callable__
const typename GridEntity< Grid, EntityDimension >::CoordinatesType&
GridEntity< Grid, EntityDimension >::getNormals() const
{
   return this->normals;
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setNormals( const CoordinatesType& normals )
{
   this->normals = normals;
   std::cout << "Setting normals to " << this->normals << std::endl;
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::getBasis() const -> CoordinatesType
{
   CoordinatesType aux = 1.0;
   return aux - this->normals;
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::IndexType
GridEntity< Grid, EntityDimension >::getOrientation() const
{
   return this->orientation;
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setOrientation( const IndexType orientation ) {
   this->orientation = orientation;
}

template< class Grid, int EntityDimension >
template< int Dimension, int... Steps, std::enable_if_t< sizeof...( Steps ) == Grid::getMeshDimension(), bool > >
__cuda_callable__
GridEntity< Grid, Dimension >
GridEntity< Grid, EntityDimension >::getNeighbourEntity() const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid, Steps... >( *this );
}

template< class Grid, int EntityDimension >
template< int Dimension,
          int Orientation,
          int... Steps,
          std::enable_if_t< sizeof...( Steps ) == Grid::getMeshDimension(), bool > >
__cuda_callable__
GridEntity< Grid, Dimension >
GridEntity< Grid, EntityDimension >::getNeighbourEntity() const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid, Orientation, Steps... >( *this );
}

template< class Grid, int EntityDimension >
template< int Dimension >
__cuda_callable__
GridEntity< Grid, Dimension >
GridEntity< Grid, EntityDimension >::getNeighbourEntity( const CoordinatesType& offset ) const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid >( *this, offset );
}

template< class Grid, int EntityDimension >
template< int Dimension, int Orientation >
__cuda_callable__
GridEntity< Grid, Dimension >
GridEntity< Grid, EntityDimension >::getNeighbourEntity( const CoordinatesType& offset ) const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid, Orientation >( *this, offset );
}

template< class Grid, int EntityDimension >
auto GridEntity< Grid, EntityDimension >::
getPoint() const -> PointType
{ 
   return this->grid.getSpaceSteps() * this->getCoordinates(); 
}

template< class Grid, int EntityDimension >
__cuda_callable__
const Grid& 
GridEntity< Grid, EntityDimension >::
getGrid() const
{
   return this->grid;
};

template< class Grid, int EntityDimension >
std::ostream& operator<<( std::ostream& str, const GridEntity< Grid, EntityDimension >& entity )
{
   str << "Entity dimension = " << EntityDimension << " coordinates = " << entity.getCoordinates() << " normals = " << entity.getNormals()
       << " index = " << entity.getIndex() << " orientation = " << entity.getOrientation();
   return str;
}

}  // namespace Meshes
}  // namespace TNL
