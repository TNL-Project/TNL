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
__cuda_callable__
inline const typename GridEntity< Grid, EntityDimension >::Coordinate&
GridEntity< Grid, EntityDimension >::getCoordinates() const
{
   return this->coordinates;
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline typename GridEntity< Grid, EntityDimension >::Coordinate&
GridEntity< Grid, EntityDimension >::getCoordinates()
{
   return this->coordinates;
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline void
GridEntity< Grid, EntityDimension >::setCoordinates( const Coordinate& coordinates )
{
   this->coordinates = coordinates;
   refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline void
GridEntity< Grid, EntityDimension >::refresh()
{
   this->index = this->grid.getEntityIndex( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline typename GridEntity< Grid, EntityDimension >::Index
GridEntity< Grid, EntityDimension >::getIndex() const
{
   TNL_ASSERT_GE( this->index, 0, "Entity index is not non-negative." );
   TNL_ASSERT_LT( this->index, grid.template getEntitiesCount< EntityDimension >(), "Entity index is out of bounds." );
   TNL_ASSERT_EQ( this->index, grid.getEntityIndex( *this ), "Wrong value of stored index." );

   return this->index;
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline bool
GridEntity< Grid, EntityDimension >::isBoundary() const
{
   return BoundaryGridEntityChecker< GridEntity >::isBoundaryEntity( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline const typename GridEntity< Grid, EntityDimension >::Point
GridEntity< Grid, EntityDimension >::getCenter() const
{
   return GridEntityCenterGetter< GridEntity >::getEntityCenter( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline typename GridEntity< Grid, EntityDimension >::Real
GridEntity< Grid, EntityDimension >::getMeasure() const
{
   return GridEntityMeasureGetter< Grid, EntityDimension >::getMeasure( this->getMesh(), *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline const Grid&
GridEntity< Grid, EntityDimension >::getMesh() const
{
   return this->grid;
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline typename GridEntity< Grid, EntityDimension >::Coordinate
GridEntity< Grid, EntityDimension >::getBasis() const
{
   return this->basis;
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline void
GridEntity< Grid, EntityDimension >::setBasis( const Coordinate& basis )
{
   this->basis = basis;
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline typename GridEntity< Grid, EntityDimension >::Index
GridEntity< Grid, EntityDimension >::getOrientation() const
{
   return this->orientation;
}

template< class Grid, int EntityDimension >
__cuda_callable__
inline void
GridEntity< Grid, EntityDimension >::setOrientation( const Index orientation ) {
   this->orientation = orientation;
}

template< class Grid, int EntityDimension >
template< int Dimension, int... Steps, std::enable_if_t< sizeof...( Steps ) == Grid::getMeshDimension(), bool > >
__cuda_callable__
inline GridEntity< Grid, Dimension >
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
inline GridEntity< Grid, Dimension >
GridEntity< Grid, EntityDimension >::getNeighbourEntity() const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid, Orientation, Steps... >( *this );
}

template< class Grid, int EntityDimension >
template< int Dimension >
__cuda_callable__
inline GridEntity< Grid, Dimension >
GridEntity< Grid, EntityDimension >::getNeighbourEntity( const Coordinate& offset ) const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid >( *this, offset );
}

template< class Grid, int EntityDimension >
template< int Dimension, int Orientation >
__cuda_callable__
inline GridEntity< Grid, Dimension >
GridEntity< Grid, EntityDimension >::getNeighbourEntity( const Coordinate& offset ) const
{
   using Getter = NeighbourGridEntityGetter< meshDimension, entityDimension, Dimension >;

   return Getter::template getEntity< Grid, Orientation >( *this, offset );
}

}  // namespace Meshes
}  // namespace TNL
