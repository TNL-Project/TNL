// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridDetails/BoundaryGridEntityChecker.h>
#include <TNL/Meshes/GridDetails/GridEntityCenterGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>
#include <TNL/Meshes/GridEntity.h>

namespace TNL {
namespace Meshes {

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
GridEntity( const Meshes::Grid< Dimension, Real, Device, Index >& grid )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( 0 ),
  orientation( 0 ){
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
GridEntity( const Meshes::Grid< Dimension, Real, Device, Index >& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates ),
  orientation( orientation ){
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
__cuda_callable__
inline typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::getCoordinates()
{
   return this->coordinates;
}

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
__cuda_callable__
inline void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::setCoordinates(
   const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
__cuda_callable__
inline void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::refresh()
{
   this->entityIndex = this->grid.getEntityIndex( *this );
}

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
__cuda_callable__
inline Index
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::getIndex() const
{
   using GridType = Meshes::Grid< Dimension, Real, Device, Index >;
   using EntityType = typename GridType::template EntityType< EntityDimension >;
   TNL_ASSERT_GE( this->entityIndex, 0, "Entity index is not non-negative." );
   TNL_ASSERT_LT( this->entityIndex, grid.template getEntitiesCount< EntityDimension >(), "Entity index is out of bounds." );
   TNL_ASSERT_EQ( this->entityIndex, grid.getEntityIndex( *this ), "Wrong value of stored index." );
   return this->entityIndex;
}

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
__cuda_callable__
inline const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
   EntityOrientationType&
   GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::getOrientation() const
{
   return this->orientation;
}

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
__cuda_callable__
inline void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::setOrientation(
   const EntityOrientationType& orientation )
{
   this->orientation = orientation;
}


template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
bool
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
isBoundaryEntity() const
{
   return BoundaryGridEntityChecker< GridEntity >::isBoundaryEntity( *this );
}

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
__cuda_callable__
inline typename Meshes::Grid< Dimension, Real, Device, Index >::PointType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::getCenter() const
{
   return GridEntityCenterGetter< GridEntity >::getEntityCenter( *this );
}

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
__cuda_callable__
inline const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::RealType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::getMeasure() const
{
   return GridEntityMeasureGetter< GridType, EntityDimension >::getMeasure( this->getMesh(), *this );
}

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
__cuda_callable__
inline const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::GridType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::getMesh() const
{
   return this->grid;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
GridEntity( const GridType& grid )
: grid( grid ),
  entityIndex( -1 )
{
   this->coordinates = CoordinatesType( (Index) 0 );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
GridEntity( const GridType& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation)
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates )
{
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::getCoordinates() const
{
   return this->coordinates;
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::getCoordinates()
{
   return this->coordinates;
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::setCoordinates(
   const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::refresh()
{
   this->entityIndex = this->grid.getEntityIndex( *this );
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline Index
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::getIndex() const
{
   TNL_ASSERT_GE( this->entityIndex, 0, "Entity index is not non-negative." );
   TNL_ASSERT_LT( this->entityIndex, grid.template getEntitiesCount< Dimension >(), "Entity index is out of bounds." );
   TNL_ASSERT_EQ( this->entityIndex, grid.getEntityIndex( *this ), "Wrong value of stored index." );
   return this->entityIndex;
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::EntityOrientationType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::getOrientation() const
{
   return EntityOrientationType( (IndexType) 0 );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
bool
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
isBoundaryEntity() const
{
   return BoundaryGridEntityChecker< GridEntity >::isBoundaryEntity( *this );
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline typename Meshes::Grid< Dimension, Real, Device, Index >::PointType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::getCenter() const
{
   return GridEntityCenterGetter< GridEntity >::getEntityCenter( *this );
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::RealType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::getMeasure() const
{
   return this->getMesh().getCellMeasure();
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::PointType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::getEntityProportions() const
{
   return grid.getSpaceSteps();
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::GridType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::getMesh() const
{
   return this->grid;
}

/****
 * Specialization for vertices
 */
template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
GridEntity( const GridType& grid )
 : grid( grid ),
   entityIndex( -1 ),
   coordinates( 0 )
{
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
GridEntity( const GridType& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation)
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates )
{
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::getCoordinates()
{
   return this->coordinates;
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::refresh()
{
   this->entityIndex = this->grid.getEntityIndex( *this );
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline Index
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::getIndex() const
{
   using GridType = Meshes::Grid< Dimension, Real, Device, Index >;
   using Vertex = typename GridType::Vertex;
   TNL_ASSERT_GE( this->entityIndex, 0, "Entity index is not non-negative." );
   TNL_ASSERT_LT( this->entityIndex, grid.template getEntitiesCount< 0 >(), "Entity index is out of bounds." );
   TNL_ASSERT_EQ( this->entityIndex, grid.getEntityIndex( *this ), "Wrong value of stored index." );
   return this->entityIndex;
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::EntityOrientationType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::getOrientation() const
{
   return EntityOrientationType( (IndexType) 0 );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
bool
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
isBoundaryEntity() const
{
   return BoundaryGridEntityChecker< GridEntity >::isBoundaryEntity( *this );
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline typename Meshes::Grid< Dimension, Real, Device, Index >::PointType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::getCenter() const
{
   return GridEntityCenterGetter< GridEntity >::getEntityCenter( *this );
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline typename Meshes::Grid< Dimension, Real, Device, Index >::PointType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::getPoint() const
{
   return getCenter();
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::RealType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::getMeasure() const
{
   return 0.0;
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::PointType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::getEntityProportions() const
{
   return PointType( 0.0 );
}

template< int Dimension, typename Real, typename Device, typename Index, typename Config >
__cuda_callable__
inline const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::GridType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::getMesh() const
{
   return this->grid;
}

}  // namespace Meshes
}  // namespace TNL
