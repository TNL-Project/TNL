// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter_impl.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter2D_impl.h>
#include <TNL/String.h>
#include <TNL/Meshes/NDimGrid.h>

#include <fstream>
#include <iomanip>

namespace TNL {
namespace Meshes {

template <typename Real, typename Device, typename Index>
Grid<2, Real, Device, Index>::Grid(): NDimGrid<2, Real, Device, Index>() {
   this->setDimensions(0);
}

template <typename Real, typename Device, typename Index>
Grid<2, Real, Device, Index>::Grid(const Index xSize, const Index ySize): NDimGrid<2, Real, Device, Index>() {
   this->setDimensions(xSize, ySize);
}

template <typename Real, typename Device, typename Index>
template <typename Entity>
__cuda_callable__ inline Entity Grid<2, Real, Device, Index>::getEntity(const IndexType& entityIndex) const {
   static_assert(Entity::getEntityDimension() <= 2 && Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions.");

   return GridEntityGetter<Grid, Entity>::getEntity(*this, entityIndex);
}

template <typename Real, typename Device, typename Index>
template <typename Entity>
__cuda_callable__ inline Index Grid<2, Real, Device, Index>::getEntityIndex(const Entity& entity) const {
   static_assert(Entity::getEntityDimension() <= 2 && Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions.");

   return GridEntityGetter<Grid, Entity>::getEntityIndex(*this, entity);
}

template <typename Real, typename Device, typename Index>
__cuda_callable__ const Real& Grid<2, Real, Device, Index>::getCellMeasure() const {
   return this->template getSpaceStepsProducts<1, 1>();
}

template <typename Real, typename Device, typename Index>
template <int EntityDimension, typename Func, typename... FuncArgs>
void Grid<2, Real, Device, Index>::forAll(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 2, "Entity dimension must be in range [0..<2]");

   auto outer = [=] __cuda_callable__(Index i, Index j, const Grid<2, Real, Device, Index>& grid, FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid, CoordinatesType(i, j));
      entity.refresh();

      func(entity, args...);
   };

   switch (EntityDimension) {
      case 0:
         this -> forEach({ 0, 0 }, { this -> dimensions.x() + 1, this -> dimensions.y() + 1 }, outer, *this, args...);
         break;
      case 1: {
         auto outerOriented = [=] __cuda_callable__(Index i, Index j, const Grid<2, Real, Device, Index>& grid, const CoordinatesType& orientation,
                                                    FuncArgs... args) mutable {
            EntityType<EntityDimension> entity(grid, CoordinatesType(i, j), orientation);

            entity.refresh();

            func(entity, args...);
         };

         this -> forEach({ 0, 0 }, { this -> dimensions.x() + 1, this -> dimensions.y() }, outerOriented, *this, CoordinatesType(1, 0), args...);
         this -> forEach({ 0, 0 }, { this -> dimensions.x(), this -> dimensions.y() + 1 }, outerOriented, *this, CoordinatesType(0, 1), args...);
         break;
      }
      case 2:
         this -> forEach({ 0, 0 }, { this -> dimensions.x(), this -> dimensions.y() }, outer, *this, args...);

         // TODO: - Verify for distributed grids
         // TNL::Algorithms::ParallelFor2D<Device>::exec(localBegin.x(), localBegin.y(), localEnd.x(), localEnd.y(), outer, * this, args...);
         break;
      default:
         break;
   }
}

template <typename Real, typename Device, typename Index>
template <int EntityDimension, typename Func, typename... FuncArgs>
void Grid<2, Real, Device, Index>::forInterior(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 2, "Entity dimension must be in range [0..<2]");

   auto outer = [=] __cuda_callable__(Index i, Index j, const Grid<2, Real, Device, Index>& grid, FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid);

      entity.setCoordinates({i, j});
      entity.refresh();

      func(entity, args...);
   };

   switch (EntityDimension) {
      case 0:
         this -> forEach({ 1, 1 }, { this -> dimensions.x(), this -> dimensions.y() }, outer, *this, args...);
         break;
      case 1: {
         auto outerOriented = [=] __cuda_callable__(Index i, Index j, Grid<2, Real, Device, Index> & grid, const CoordinatesType& orientation,
                                                    FuncArgs... args) mutable {
            EntityType<EntityDimension> entity(grid, CoordinatesType(i, j), orientation);

            entity.refresh();

            func(entity, args...);
         };

         this -> forEach({ 1, 0 }, { this -> dimensions.x(), this -> dimensions.y() }, outerOriented, *this, CoordinatesType(1, 0), args...);
         this -> forEach({ 0, 1 }, { this -> dimensions.x(), this -> dimensions.y() }, outerOriented, *this, CoordinatesType(0, 1), args...);
         break;
      }
      case 2:
         this -> forEach({ 1, 1 }, { this -> dimensions.x() - 1, this -> dimensions.y() - 1 }, outer, *this, args...);
         // TODO: - Verify for distributed grids
         // TNL::Algorithms::ParallelFor2D<Device>::exec(interiorBegin.x(), interiorBegin.y(), interiorEnd.x(), interiorEnd.y(), outer, *this,
         // args...);
         break;
      default:
         break;
   }
}

template <typename Real, typename Device, typename Index>
template <int EntityDimension, typename Func, typename... FuncArgs>
void Grid<2, Real, Device, Index>::forBoundary(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 2, "Entity dimension must be in range [0...2]");

   auto outer = [=] __cuda_callable__(Index i, Index axis, Index axisIndex, const Grid<2, Real, Device, Index>& grid, FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid);

      switch (axis) {
         case 0:
            entity.setCoordinates({axisIndex, i});
            break;
         case 1:
            entity.setCoordinates({i, axisIndex});
            break;
         default:
            TNL_ASSERT_TRUE(false, "Received axis index. Expect in range [0..<1]");
      }

      entity.refresh();

      func(entity, args...);
   };

   switch (EntityDimension) {
      case 0:
         // Lower horizontal
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.x() + 1, outer, 0, 0, *this, args...);
         // Upper horizontal
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.x() + 1, outer, 0, this->dimensions.y(), *this, args...);
         // Left vertical
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.y() + 1, outer, 1, 0, *this, args...);
         // Right vertical
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.y() + 1, outer, 1, this->dimensions.x(), *this, args...);
         break;
      case 1: {
         auto outerOriented = [=] __cuda_callable__(Index i, Index axis, Index axisIndex, const Grid<2, Real, Device, Index>& grid,
                                                    const CoordinatesType& orientation, FuncArgs... args) mutable {
            CoordinatesType coordinates;

            switch (axis) {
               case 0:
                  coordinates[0] = axisIndex;
                  coordinates[1] = i;
                  break;
               case 1:
                  coordinates[0] = i;
                  coordinates[1] = axisIndex;
                  break;
               default:
                  TNL_ASSERT_TRUE(false, "Received axis index. Expect in range [0...1]");
            }

            EntityType<EntityDimension> entity(grid, coordinates, orientation);

            entity.refresh();

            func(entity, args...);
         };

         // Lower horizontal
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.x(), outerOriented, 0, 0, *this, CoordinatesType(1, 0), args...);
         // Upper horizontal
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.x(), outerOriented, 0, this->dimensions.y(), *this, CoordinatesType(1, 0),
                                                    args...);
         // Left vertical
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.y(), outerOriented, 1, 0, *this, CoordinatesType(0, 1), args...);
         // Right vertical
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.y(), outerOriented, 1, this->dimensions.x(), *this, CoordinatesType(0, 1),
                                                    args...);
         break;
      }
      case 2:
         // Lower horizontal
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.x(), outer, 0, 0, *this, args...);
         // Upper horizontal
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.x(), outer, 0, this->dimensions.y() - 1, *this, args...);
         // Left vertical
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.y(), outer, 1, 0, *this, args...);
         // Right vertical
         TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.y(), outer, 1, this->dimensions.x() - 1, *this, args...);
         // TODO: - Verify with the distributed grid
         /*if (localBegin < interiorBegin && interiorEnd < localEnd) {
            // Lower horizontal
            TNL::Algorithms::ParallelFor2D<Device>::exec(interiorBegin.x() - 1, interiorBegin.y() - 1, interiorEnd.x() + 1, interiorBegin.y() + 1,
         outer, *this, args...);
            // Upper horizontal
            TNL::Algorithms::ParallelFor2D<Device>::exec(interiorBegin.x() - 1, interiorEnd.y() - 1, interiorEnd.x() + 1, interiorEnd.y() + 1, outer,
         *this, args...);
            // Left vertical
            TNL::Algorithms::ParallelFor2D<Device>::exec(interiorBegin.x() -1, interiorBegin.y() - 1, interiorBegin.x() + 1, interiorEnd.y() + 1,
         outer, *this, args...);
            // Right vertical
            TNL::Algorithms::ParallelFor2D<Device>::exec(interiorEnd.x() - 1, interiorBegin.y() - 1, interiorEnd.x() + 1, interiorEnd.y() + 1, outer,
         *this, args...); return;
         }

         // Lower horizontal
         TNL::Algorithms::ParallelFor2D<Device>::exec(localBegin.x(), localBegin.y(), localEnd.x(), interiorBegin.y(), outer, *this, args...);
         // Upper horizontal
         TNL::Algorithms::ParallelFor2D<Device>::exec(localBegin.x(), interiorEnd.y(), localEnd.x(), localEnd.y(), outer, *this, args...);
         // Left vertical
         TNL::Algorithms::ParallelFor2D<Device>::exec(localBegin.x(), interiorBegin.y(), interiorBegin.x(), interiorEnd.y(), outer, *this, args...);
         // Right vertical
         TNL::Algorithms::ParallelFor2D<Device>::exec(interiorEnd.x(), interiorBegin.y(), localEnd.x(), interiorEnd.y(), outer, *this, args...);*/
         break;
      default:
         break;
   }
}

}  // namespace Meshes
}  // namespace TNL
