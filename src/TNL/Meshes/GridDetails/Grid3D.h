// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Logger.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridEntityConfig.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index >
class Grid< 3, Real, Device, Index >: public NDimGrid<3, Real, Device, Index>
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using GlobalIndexType = Index;
   using PointType = Containers::StaticVector< 3, Real >;
   using CoordinatesType = Containers::StaticVector< 3, Index >;

   // TODO: deprecated and to be removed (GlobalIndexType shall be used instead)
   using IndexType = Index;

   template <int EntityDimension>
   using EntityType = GridEntity<Grid, EntityDimension>;

   using Base = NDimGrid<3, Real, Device, Index>;
   using Coordinate = typename Base::Coordinate;
   using Point = typename Base::Point;
   using EntitiesCounts = typename Base::EntitiesCounts;

   static constexpr int getMeshDimension() { return 3; };

   /**
    * \brief See Grid1D::Grid().
    */
   Grid() = default;

   Grid( Index xSize, Index ySize, Index zSize );

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
   template< typename Entity >
   __cuda_callable__
   inline Entity
   getEntity( const IndexType& entityIndex ) const;

   /**
    * \brief See Grid1D::getEntityIndex().
    */
   template< typename Entity >
   __cuda_callable__
   inline Index
   getEntityIndex( const Entity& entity ) const;

   /**
    * \breif Returns the measure (volume) of a cell in this grid.
    */
   __cuda_callable__
   inline const RealType&
   getCellMeasure() const;

   /**
    * \brief Traverses all elements
    */
   template<int EntityDimension, typename Func, typename... FuncArgs>
   void forAll(Func func, FuncArgs... args) const;

   /**
    * \brief Traversers interior elements
    */
   template<int EntityDimension, typename Func, typename... FuncArgs>
   void forInterior(Func func, FuncArgs... args) const;

   /**
    * \brief Traversers boundary elements
    */
   template<int EntityDimension, typename Func, typename... FuncArgs>
   void forBoundary(Func func, FuncArgs... args) const;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Grid3D_impl.h>
