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
class Grid<3, Real, Device, Index> : public NDimGrid<3, Real, Device, Index> {
   public:
      template <int EntityDimension>
      using EntityType = GridEntity<Grid, EntityDimension>;

      using Base = NDimGrid<3, Real, Device, Index>;
      using Coordinate = typename Base::Coordinate;
      using Point = typename Base::Point;
      using EntitiesCounts = typename Base::EntitiesCounts;

      /**
       * \brief See Grid1D::Grid().
       */
      Grid();

      Grid(const Index xSize, const Index ySize, const Index zSize);

      // /**
      //  * \brief Gets number of entities in this grid.
      //  * \tparam EntityDimension Integer specifying dimension of the entity.
      //  */
      // template< int EntityDimension >
      // __cuda_callable__
      // IndexType getEntitiesCount() const;

      /**
       * \brief Gets number of entities in this grid.
       * \tparam Entity Type of the entity.
      //  */
      // template< typename Entity >
      // __cuda_callable__
      // IndexType getEntitiesCount() const;

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
       * \breif Returns the measure (volume) of a cell in this grid.
       */
      __cuda_callable__ inline
      const Real& getCellMeasure() const;

      template <int EntityDimension, typename Func, typename... FuncArgs>
      inline void forAll(Func func, FuncArgs... args) const;

      template <int EntityDimension, typename Func, typename... FuncArgs>
      inline void forAll(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const;

      template <int EntityDimension, typename Func, typename... FuncArgs>
      inline void forInterior(Func func, FuncArgs... args) const;

      template <int EntityDimension, typename Func, typename... FuncArgs>
      inline void forInterior(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const;

      template <int EntityDimension, typename Func, typename... FuncArgs>
      inline void forBoundary(Func func, FuncArgs... args) const;

      template <int EntityDimension, typename Func, typename... FuncArgs>
      inline void forBoundary(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/Grid3D.hpp>
