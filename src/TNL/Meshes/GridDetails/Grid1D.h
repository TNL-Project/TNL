// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Logger.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityTopology.h>
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridEntityConfig.h>

namespace TNL {
namespace Meshes {

template <typename Real, typename Device, typename Index>
class Grid<1, Real, Device, Index> : public NDimGrid<1, Real, Device, Index> {
  public:
   template <int EntityDimension>
   using EntityType = GridEntity<Grid, EntityDimension>;

   using Base = NDimGrid<1, Real, Device, Index>;
   using Coordinate = typename Base::Coordinate;

   /**
    * \brief Basic constructor.
    */
   Grid();

   Grid(const Index xSize);

   /**
    * \brief Gets number of entities in this grid.
    * \tparam Entity Type of the entity.
    */
   // template <typename Entity>
   // __cuda_callable__ Index getEntitiesCount() const;

   /**
    * \brief Gets entity type using entity index.
    * \param entityIndex Index of entity.
    * \tparam Entity Type of the entity.
    */
   template <typename Entity>
   __cuda_callable__ inline Entity getEntity(const Index &entityIndex) const;

   /**
    * \brief Gets entity index using entity type.
    * \param entity Type of entity.
    * \tparam Entity Type of the entity.
    */
   template <typename Entity>
   __cuda_callable__ inline Index getEntityIndex(const Entity &entity) const;

   /**
    * \brief Returns the measure (length) of a cell in this grid.
    */
   __cuda_callable__ inline const Real &getCellMeasure() const;

   /*
    * @brief Traverses all elements
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   void forAll(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   void forInterior(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   void forBoundary(Func func, FuncArgs... args) const;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Grid1D_impl.h>
