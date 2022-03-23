// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace TNL {
namespace Meshes {

#define __GRID_2D_TEMPLATE__ template <typename Real, typename Device, typename Index>
#define __GRID_2D_PREFIX__ Grid<2, Real, Device, Index>

__GRID_2D_TEMPLATE__
__GRID_2D_PREFIX__::Grid() {
   this->setDimensions(0);
}

__GRID_2D_TEMPLATE__
__GRID_2D_PREFIX__::Grid(const Index xSize, const Index ySize) {
   this->setDimensions(xSize, ySize);
}

__GRID_2D_TEMPLATE__
template <typename Entity>
__cuda_callable__ inline Entity __GRID_2D_PREFIX__::getEntity(const Index& entityIndex) const {
   static_assert(Entity::entityDimension <= 2 && Entity::entityDimension >= 0, "Wrong grid entity dimensions.");

   return GridEntityGetter<Grid, Entity::entityDimension>::getEntity(*this, entityIndex);
}

__GRID_2D_TEMPLATE__
template <typename Entity>
__cuda_callable__ inline Index __GRID_2D_PREFIX__::getEntityIndex(const Entity& entity) const {
   static_assert(Entity::entityDimension <= 2 && Entity::entityDimension >= 0, "Wrong grid entity dimensions.");

   return GridEntityGetter<Grid, Entity::entityDimension>::getEntityIndex(*this, entity);
}

__GRID_2D_TEMPLATE__
__cuda_callable__ const Real& __GRID_2D_PREFIX__::getCellMeasure() const {
   return this->template getSpaceStepsProducts<1, 1>();
}

__GRID_2D_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __GRID_2D_PREFIX__::forAll(Func func, FuncArgs... args) const {
   auto exec = [=] __cuda_callable__ (const Coordinate& coordinate,
                                      const Coordinate& basis,
                                      const Grid &grid,
                                      FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid, coordinate, basis);

      entity.refresh();

      func(entity, args...);
   };

   this -> template traverseAll<EntityDimension>(exec, *this, args...);
}

__GRID_2D_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __GRID_2D_PREFIX__::forInterior(Func func, FuncArgs... args) const {
   auto exec = [=] __cuda_callable__ (const Coordinate& coordinate,
                                      const Coordinate& basis,
                                      const Grid &grid,
                                      FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid, coordinate, basis);

      entity.refresh();

      func(entity, args...);
   };

   this -> template traverseInterior<EntityDimension>(exec, *this, args...);
}

__GRID_2D_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __GRID_2D_PREFIX__::forBoundary(Func func, FuncArgs... args) const {
   auto exec = [=] __cuda_callable__(const Coordinate& coordinate,
                                     const Coordinate& basis,
                                     const Grid& grid,
                                     FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid, coordinate, basis);

      entity.refresh();

      func(entity, args...);
   };

   this->template traverseBoundary<EntityDimension>(exec, *this, args...);
}

}  // namespace Meshes
}  // namespace TNL
