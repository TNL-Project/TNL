// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridDetails/NDimGrid.h>

namespace TNL {
namespace Meshes {

template <class, int>
class GridEntity;

template <typename Real, typename Device, typename Index>
class Grid<2, Real, Device, Index> : public NDimGrid<2, Real, Device, Index> {
  public:
   template <int EntityDimension>
   using EntityType = GridEntity<Grid, EntityDimension>;

   using Base = NDimGrid<2, Real, Device, Index>;
   using Coordinate = typename Base::Coordinate;
   using Point = typename Base::Point;
   using EntitiesCounts = typename Base::EntitiesCounts;

   Grid() = default;
   Grid(const Index xSize, const Index ySize);

   /**
    * \brief See Grid1D::getEntityIndex().
    */
   template <typename Entity>
   __cuda_callable__ inline Index getEntityIndex(const Entity& entity) const;

   /**
    * @brief Traverser all elements in rect
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void forAll(Func func, FuncArgs... args) const;

   /**
    * @brief Traverser all elements in rect
    * @param from - bottom left anchor of traverse rect
    * @param to - top right anchor of traverse rect
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void forAll(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const;

   /**
    * @brief Traverser interior elements in rect
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void forInterior(Func func, FuncArgs... args) const;

   /**
    * @brief Traverser interior elements
    * @param from - bottom left anchor of traverse rect
    * @param to - top right anchor of traverse rect
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void forInterior(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const;

   /**
    * @brief Traverser boundary elements in rect
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void forBoundary(Func func, FuncArgs... args) const;

   /**
    * @brief Traverser boundary elements in rect
    * @param from - bottom left anchor of traverse rect
    * @param to - top right anchor of traverse rect
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   inline void forBoundary(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/Grid2D.hpp>
