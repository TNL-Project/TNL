// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridDetails/NDimGrid.h>

namespace TNL {
namespace Meshes {

template<class, int>
class GridEntity;

template <typename Real, typename Device, typename Index>
class Grid<2, Real, Device, Index>: public NDimGrid<2, Real, Device, Index> {
  public:
   template <int EntityDimension>
   using EntityType = GridEntity<Grid, EntityDimension>;

   using Base = NDimGrid<2, Real, Device, Index>;
   using Coordinate = typename Base::Coordinate;
   using Point = typename Base::Point;
   using EntitiesCounts = typename Base::EntitiesCounts;

   /**
    * \brief See Grid1D::Grid().
    */
   Grid() = default;

   Grid(const Index xSize, const Index ySize);

   // TODO: - Fix method
   // /**
   //  * \brief Gets number of entities in this grid.
   //  * \tparam Entity Type of the entity.
   //  */
   // template <typename Entity, std::enable_if_t<(std::is_integral<Entity>::value), bool> = true>
   // __cuda_callable__ inline Index getEntitiesCountA() const;

   /**
    * \brief See Grid1D::getEntity().
    */
   template <typename Entity>
   __cuda_callable__ inline
   Entity getEntity(const Index& entityIndex) const;

   /**
    * \brief See Grid1D::getEntityIndex().
    */
   template <typename Entity>
   __cuda_callable__ inline
   Index getEntityIndex(const Entity& entity) const;

   /**
    * \brief Returns the measure (area) of a cell in this grid.
    */
   __cuda_callable__ inline
   const Real& getCellMeasure() const;

   /*
    * @brief Traverses all elements
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline
   void forAll(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline
   void forInterior(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline
   void forBoundary(Func func, FuncArgs... args) const;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/Grid2D.hpp>
