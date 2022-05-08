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

#define __GRID_ENTITY_TEMPLATE__ template< class Grid, int EntityDimension >
#define __GRID_ENTITY_PREFIX__ GridEntity< Grid, EntityDimension >

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline const typename __GRID_ENTITY_PREFIX__::Coordinate&
__GRID_ENTITY_PREFIX__::getCoordinates() const
{
   return this->coordinates;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline typename __GRID_ENTITY_PREFIX__::Coordinate&
__GRID_ENTITY_PREFIX__::getCoordinates()
{
   return this->coordinates;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline void
__GRID_ENTITY_PREFIX__::setCoordinates( const Coordinate& coordinates )
{
   this->coordinates = coordinates;
   refresh();
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline void
__GRID_ENTITY_PREFIX__::refresh()
{
   this->index = this->grid.getEntityIndex( *this );
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline typename __GRID_ENTITY_PREFIX__::Index
__GRID_ENTITY_PREFIX__::getIndex() const
{
   TNL_ASSERT_GE( this->index, 0, "Entity index is not non-negative." );
   TNL_ASSERT_LT( this->index, grid.template getEntitiesCount< EntityDimension >(), "Entity index is out of bounds." );
   TNL_ASSERT_EQ( this->index, grid.getEntityIndex( *this ), "Wrong value of stored index." );

   return this->index;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline bool
__GRID_ENTITY_PREFIX__::isBoundary() const
{
   return BoundaryGridEntityChecker< GridEntity >::isBoundaryEntity( *this );
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline const typename __GRID_ENTITY_PREFIX__::Point
__GRID_ENTITY_PREFIX__::getCenter() const
{
   return GridEntityCenterGetter< GridEntity >::getEntityCenter( *this );
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline typename __GRID_ENTITY_PREFIX__::Real
__GRID_ENTITY_PREFIX__::getMeasure() const
{
   return GridEntityMeasureGetter< Grid, EntityDimension >::getMeasure( this->getMesh(), *this );
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline const Grid&
__GRID_ENTITY_PREFIX__::getMesh() const
{
   return this->grid;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline typename __GRID_ENTITY_PREFIX__::Coordinate
__GRID_ENTITY_PREFIX__::getBasis() const
{
   return this->basis;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline void
__GRID_ENTITY_PREFIX__::setBasis( const Coordinate& basis )
{
   this->basis = basis;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline typename __GRID_ENTITY_PREFIX__::Index
__GRID_ENTITY_PREFIX__::getOrientation() const
{
   return this->orientation;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__
inline void
__GRID_ENTITY_PREFIX__::setOrientation( const Index orientation ) {
   this->orientation = orientation;
}

__GRID_ENTITY_TEMPLATE__
template< int Dimension, int... Steps, std::enable_if_t< sizeof...( Steps ) == Grid::getMeshDimension(), bool > >
__cuda_callable__
inline GridEntity< Grid, Dimension >
__GRID_ENTITY_PREFIX__::getNeighbourEntity() const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid, Steps... >( *this );
}

__GRID_ENTITY_TEMPLATE__
template< int Dimension,
          int Orientation,
          int... Steps,
          std::enable_if_t< sizeof...( Steps ) == Grid::getMeshDimension(), bool > >
__cuda_callable__
inline GridEntity< Grid, Dimension >
__GRID_ENTITY_PREFIX__::getNeighbourEntity() const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid, Orientation, Steps... >( *this );
}

__GRID_ENTITY_TEMPLATE__
template< int Dimension >
__cuda_callable__
inline GridEntity< Grid, Dimension >
__GRID_ENTITY_PREFIX__::getNeighbourEntity( const Coordinate& offset ) const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid >( *this, offset );
}

__GRID_ENTITY_TEMPLATE__
template< int Dimension, int Orientation >
__cuda_callable__
inline GridEntity< Grid, Dimension >
__GRID_ENTITY_PREFIX__::getNeighbourEntity( const Coordinate& offset ) const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid, Orientation >( *this, offset );
}

}  // namespace Meshes
}  // namespace TNL
