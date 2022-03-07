// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter_impl.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter1D_impl.h>
#include <TNL/String.h>

#include <fstream>
#include <iomanip>

namespace TNL {
namespace Meshes {

template <typename Real, typename Device, typename Index>
Grid<1, Real, Device, Index>::Grid() {
   this->setDimensions(0);
}

template <typename Real, typename Device, typename Index>
Grid<1, Real, Device, Index>::Grid(const Index xSize) {
   this->setDimensions(xSize);
}

// template <typename Real, typename Device, typename Index>
// template <typename Entity>
// __cuda_callable__ inline Index Grid<1, Real, Device, Index>::getEntitiesCount() const {
//    return getEntitiesCount<Entity::getEntityDimension()>();
// }

template <typename Real, typename Device, typename Index>
template <typename Entity>
__cuda_callable__ inline Entity Grid<1, Real, Device, Index>::getEntity(
    const Index &entityIndex) const {
   static_assert(Entity::getEntityDimension() <= 1 && Entity::getEntityDimension() >= 0,
                 "Wrong grid entity dimensions.");

   return GridEntityGetter<Grid, Entity>::getEntity(*this, entityIndex);
}

template <typename Real, typename Device, typename Index>
template <typename Entity>
__cuda_callable__ inline Index Grid<1, Real, Device, Index>::getEntityIndex(
    const Entity &entity) const {
   static_assert(Entity::getEntityDimension() <= 1 && Entity::getEntityDimension() >= 0,
                 "Wrong grid entity dimensions.");

   return GridEntityGetter<Grid, Entity>::getEntityIndex(*this, entity);
}

template <typename Real, typename Device, typename Index>
__cuda_callable__ const Real &Grid<1, Real, Device, Index>::getCellMeasure() const {
   return this->template getSpaceStepsProducts<1>();
}

template <typename Real, typename Device, typename Index>
template <int EntityDimension, typename Func, typename... FuncArgs>
void Grid<1, Real, Device, Index>::forAll(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 1,
                 "Entity dimension must be either 0 or 1");

   auto outer = [=] __cuda_callable__(Index i, const Grid<1, Real, Device, Index> &grid,
                                      FuncArgs... args) mutable {
      // EntityType<EntityDimension> entity(grid);

      // entity.setCoordinates(i);
      // entity.refresh();

      // func(entity, args...);
   };

   switch (EntityDimension) {
      case 0:
         this -> forEach({ 0 }, { this -> dimensions.x() + 1 }, outer, *this, args...);
         break;
      case 1:
         // TODO: - Update for distributed grid
         // TNL::Algorithms::ParallelFor<Device>::exec(localBegin.x(), localEnd.x(), outer, *this,
         // args...);

         this -> forEach({ 0 }, this -> dimensions, outer, *this, args...);
         break;
      default:
         break;
   }
}

template <typename Real, typename Device, typename Index>
template <int EntityDimension, typename Func, typename... FuncArgs>
void Grid<1, Real, Device, Index>::forBoundary(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 1,
                 "Entity dimension must be either 0 or 1");

   auto outer = [=] __cuda_callable__(Index i, const Grid<1, Real, Device, Index> &grid,
                                      FuncArgs... args) mutable {
      // EntityType<EntityDimension> entity(grid);

      // entity.setCoordinates(i);
      // entity.refresh();

      // func(entity, args...);
   };

   switch (EntityDimension) {
      case 0:
         this -> forEach({ 0 }, { 1 }, outer, *this, args...);
         this -> forEach({ this -> dimensions.x() }, { this -> dimensions.x() + 1 }, outer, *this, args...);
         break;
      case 1:
         this -> forEach({ 0 }, { 1 }, outer, *this, args...);
         this -> forEach({ this -> dimensions.x() - 1 }, { this -> dimensions.x() }, outer, *this, args...);
         // TODO: - Verify for distributed grid
         /*if (localBegin < interiorBegin && interiorEnd < localEnd) {
            outer(interiorBegin.x() - 1, *this, args...);
            outer(interiorEnd.x(), *this, args...);
            break;
         }

         if (localBegin < interiorBegin) {
            outer(interiorBegin.x() - 1, *this, args...);
            break;
         }

         if (interiorEnd < localEnd)
            outer(interiorEnd.x(), *this, args...);*/
         break;
      default:
         break;
   }
}

template <typename Real, typename Device, typename Index>
template <int EntityDimension, typename Func, typename... FuncArgs>
void Grid<1, Real, Device, Index>::forInterior(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 1,
                 "Entity dimension must be either 0 or 1");

   auto outer = [=] __cuda_callable__(Index i, const Grid<1, Real, Device, Index> &grid,
                                      FuncArgs... args) mutable {
      // EntityType<EntityDimension> entity(grid);

      // entity.setCoordinates(i);
      // entity.refresh();

      // func(entity, args...);
   };

   switch (EntityDimension) {
      case 0:
         this -> forEach({ 1 }, this -> dimensions, outer, *this, args...);
         break;
      case 1:
         this -> forEach({ 1 }, { this -> dimensions.x() - 1 }, outer, *this, args...);
         // TODO: - Verify for distributed grids
         // TNL::Algorithms::ParallelFor<Device>::exec(interiorBegin.x(), interiorEnd.x(), outer,
         // *this, args...);
         break;
      default:
         break;
   }
}

}  // namespace Meshes
}  // namespace TNL
