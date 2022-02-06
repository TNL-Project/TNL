// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Logger.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityTopology.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridEntityConfig.h>

namespace TNL {
namespace Meshes {

template <typename Real, typename Device, typename Index>
class Grid<2, Real, Device, Index>: public NDimGrid<2, Real, Device, Index> {
  public:
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index GlobalIndexType;
   typedef Containers::StaticVector<2, Real> PointType;
   typedef Containers::StaticVector<2, Index> CoordinatesType;

   // TODO: deprecated and to be removed (GlobalIndexType shall be used instead)
   using IndexType = Index;

   static constexpr int
   getMeshDimension()
   {
      return 2;
   };

   template <int EntityDimension, typename Config = GridEntityCrossStencilStorage<1> >
   using EntityType = GridEntity<Grid, EntityDimension, Config>;

   typedef EntityType<getMeshDimension(), GridEntityCrossStencilStorage<1> > Cell;
   typedef EntityType<getMeshDimension() - 1> Face;
   typedef EntityType<0> Vertex;

   /**
    * \brief See Grid1D::Grid().
    */
   Grid() = default;

   Grid(const Index xSize, const Index ySize);

   // empty destructor is needed only to avoid crappy nvcc warnings
   ~Grid() = default;

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
   __cuda_callable__ inline Entity getEntity(const Index& entityIndex) const;

   /**
    * \brief See Grid1D::getEntityIndex().
    */
   template <typename Entity>
   __cuda_callable__ inline Index getEntityIndex(const Entity& entity) const;

   /**
    * \breif Returns the measure (area) of a cell in this grid.
    */
   __cuda_callable__ inline const Real& getCellMeasure() const;

   /*
    * @brief Traverses all elements
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   void forAll(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   void forInterior(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   void forBoundary(Func func, FuncArgs... args) const;

  protected:
   template <typename, typename, int>
   friend class GridEntityGetter;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Grid2D_impl.h>
