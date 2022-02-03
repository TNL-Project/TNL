// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Logger.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridEntityTopology.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridEntityConfig.h>
#include <TNL/Algorithms/ParallelFor.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index>
class Grid< 1, Real, Device, Index >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using GlobalIndexType = Index;
   using PointType = Containers::StaticVector< 1, Real >;
   using CoordinatesType = Containers::StaticVector< 1, Index >;

   // TODO: deprecated and to be removed (GlobalIndexType shall be used instead)
   using IndexType = Index;

   template< int EntityDimension,
             typename Config = GridEntityCrossStencilStorage< 1 > >
   using EntityType = GridEntity< Grid, EntityDimension, Config >;

   typedef EntityType< 1, GridEntityCrossStencilStorage< 1 > > Cell;
   typedef EntityType< 0 > Face;
   typedef EntityType< 0 > Vertex;

   /**
    * \brief Basic constructor.
    */
   Grid() = default;

   Grid( Index xSize );

   // empty destructor is needed only to avoid crappy nvcc warnings
   ~Grid() = default;

   /**
    * \brief Sets the origin and proportions of this grid.
    * \param origin Point where this grid starts.
    * \param proportions Total length of this grid.
    */
   void
   setDomain( const PointType& origin, const PointType& proportions );

   /**
    * \brief Gets number of entities in this grid.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   IndexType
   getEntitiesCount() const;

   /**
    * \brief Gets entity type using entity index.
    * \param entityIndex Index of entity.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   inline Entity
   getEntity( const IndexType& entityIndex ) const;

   /**
    * \brief Gets entity index using entity type.
    * \param entity Type of entity.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   inline Index
   getEntityIndex( const Entity& entity ) const;

   /**
    * \brief Returns product of space steps to the xPow.
    * \tparam xPow Exponent.
    */
   template< int xPow >
   __cuda_callable__
   const RealType&
   getSpaceStepsProducts() const;

   /**
    * \brief Returns the measure (length) of a cell in this grid.
    */
   __cuda_callable__
   inline const RealType&
   getCellMeasure() const;

    /*
    * @brief Traverses all elements
    */
   template <int EntityDimension, typename Func, typename... FuncArgs>
   void forAll(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   void forInterior(Func func, FuncArgs... args) const;

   template <int EntityDimension, typename Func, typename... FuncArgs>
   void forBoundary(Func func, FuncArgs... args) const;

   void writeProlog( Logger& logger ) const;

protected:
   void
   computeProportions();

   void computeSpaceStepPowers();

   void computeSpaceSteps();

   PointType spaceSteps;

   RealType spaceStepsProducts[ 5 ];
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Grid1D_impl.h>
