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

namespace TNL {
namespace Meshes {

#define __GRID_ENTITY_TEMPLATE__ template <typename Grid, int EntityDimension>
#define __GRID_ENTITY_PREFIX__ GridEntity<Grid, EntityDimension>

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
const typename __GRID_ENTITY_PREFIX__::Coordinate& __GRID_ENTITY_PREFIX__::getCoordinates() const {
   return this -> coordinates;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
typename __GRID_ENTITY_PREFIX__::Coordinate& __GRID_ENTITY_PREFIX__::getCoordinates() {
   return this -> coordinates;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
void __GRID_ENTITY_PREFIX__::setCoordinates(const Coordinate& coordinates) {
   this -> coordinates = coordinates;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
void __GRID_ENTITY_PREFIX__::refresh() {
   this -> entityIndex = this -> grid.getEntityIndex( *this );
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
typename __GRID_ENTITY_PREFIX__::Index __GRID_ENTITY_PREFIX__::getIndex() const {
   TNL_ASSERT_GE( this->entityIndex, 0, "Entity index is not non-negative." );
   TNL_ASSERT_LT( this->entityIndex, grid.template getEntitiesCount< EntityDimension >(), "Entity index is out of bounds." );
   TNL_ASSERT_EQ( this->entityIndex, grid.getEntityIndex( *this ), "Wrong value of stored index." );

   return this->entityIndex;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
bool __GRID_ENTITY_PREFIX__::isBoundary() const {
   return BoundaryGridEntityChecker<GridEntity>::isBoundaryEntity(*this);
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
const typename __GRID_ENTITY_PREFIX__::Point& __GRID_ENTITY_PREFIX__::getCenter() const {
   return GridEntityCenterGetter<GridEntity>::getEntityCenter(*this);
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
typename __GRID_ENTITY_PREFIX__::Real __GRID_ENTITY_PREFIX__::getMeasure() const {
   return GridEntityMeasureGetter<Grid, EntityDimension>::getMeasure( this->getMesh(), *this );
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
const Grid& __GRID_ENTITY_PREFIX__::getMesh() const {
   return this->grid;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
const typename __GRID_ENTITY_PREFIX__::Coordinate& __GRID_ENTITY_PREFIX__::getBasis() const {
   return this -> basis;
}

__GRID_ENTITY_TEMPLATE__
__cuda_callable__ inline
void __GRID_ENTITY_PREFIX__::setBasis(const Coordinate& basis) {
   this -> basis = basis;
}

} // namespace Meshes
} // namespace TNL
