// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <iomanip>
#include <TNL/Assert.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter_impl.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter1D_impl.h>
#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index >
Grid< 1, Real, Device, Index >::Grid() {
   this->setDimensions(0);
}

template< typename Real,
          typename Device,
          typename Index >
Grid< 1, Real, Device, Index >::Grid( const Index xSize ) {
   this->setDimensions(xSize);
}

template< typename Real, typename Device, typename Index >
void
Grid< 1, Real, Device, Index >::computeSpaceSteps()
{
   if( this->getDimensions().x() != 0 ) {
      this->spaceSteps.x() = this->proportions.x() / (Real) this->getDimensions().x();
      this->computeSpaceStepPowers();
   }
}

template< typename Real, typename Device, typename Index >
void
Grid< 1, Real, Device, Index >::computeSpaceStepPowers()
{
   const RealType& hx = this->spaceSteps.x();
   this->spaceStepsProducts[ 0 ] = 1.0 / ( hx * hx );
   this->spaceStepsProducts[ 1 ] = 1.0 / hx;
   this->spaceStepsProducts[ 2 ] = 1.0;
   this->spaceStepsProducts[ 3 ] = hx;
   this->spaceStepsProducts[ 4 ] = hx * hx;
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 1, Real, Device, Index > ::computeProportions()
{
   this->proportions.x() = this->dimensions.x() * this->spaceSteps.x();
}


// template< typename Real,
//           typename Device,
//           typename Index  >
// void Grid< 1, Real, Device, Index >::setDimensions( const Index xSize )
// {
//    this->setDimensions(xSize);

//    computeSpaceSteps();

//    // only default behaviour, DistributedGrid must use the setters explicitly after setDimensions
//    localBegin = 0;
//    interiorBegin = 1;
//    localEnd = this -> dimensions;
//    interiorEnd = this -> dimensions - 1;
// }

template< typename Real, typename Device, typename Index >
void
Grid< 1, Real, Device, Index >::setDomain( const PointType& origin, const PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 1, Real, Device, Index >::PointType&
   Grid< 1, Real, Device, Index >::getProportions() const
{
   return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Entity >
__cuda_callable__  inline
Index
Grid< 1, Real, Device, Index >::
getEntitiesCount() const
{
   return getEntitiesCount< Entity::getEntityDimension() >();
}

template< typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
inline Entity
Grid< 1, Real, Device, Index >::getEntity( const IndexType& entityIndex ) const
{
   static_assert( Entity::getEntityDimension() <= 1 && Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntity( *this, entityIndex );
}

template< typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
inline Index
Grid< 1, Real, Device, Index >::getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::getEntityDimension() <= 1 && Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntityIndex( *this, entity );
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
inline const typename Grid< 1, Real, Device, Index >::PointType&
Grid< 1, Real, Device, Index >::getSpaceSteps() const
{
   return this->spaceSteps;
}

template< typename Real, typename Device, typename Index >
inline void
Grid< 1, Real, Device, Index >::setSpaceSteps( const typename Grid< 1, Real, Device, Index >::PointType& steps )
{
   this->spaceSteps = steps;
   computeSpaceStepPowers();
   computeProportions();
}

template< typename Real, typename Device, typename Index >
template< int xPow >
__cuda_callable__
inline const Real&
Grid< 1, Real, Device, Index >::getSpaceStepsProducts() const
{
   static_assert( xPow >= -2 && xPow <= 2, "unsupported value of xPow" );
   return this->spaceStepsProducts[ xPow + 2 ];
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
const Real&
Grid< 1, Real, Device, Index >::getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1 >();
}

template< typename Real,
          typename Device,
          typename Index >
template<int EntityDimension, typename Func, typename... FuncArgs>
void Grid<1, Real, Device, Index>::forAll(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 1, "Entity dimension must be either 0 or 1");

   auto outer = [=] __cuda_callable__(Index i, const Grid<1, Real, Device, Index>&grid, FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid);

      entity.setCoordinates(i);
      entity.refresh();

      func(entity, args...);
   };

   switch (EntityDimension) {
   case 0:
      TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.x() + 1, outer, *this, args...);
      break;
   case 1:
      //TODO: - Update for distributed grid
      //TNL::Algorithms::ParallelFor<Device>::exec(localBegin.x(), localEnd.x(), outer, *this, args...);

      TNL::Algorithms::ParallelFor<Device>::exec(0, this->dimensions.x(), outer, *this, args...);
      break;
   default: break;
   }
}

template< typename Real,
          typename Device,
          typename Index >
template<int EntityDimension, typename Func, typename... FuncArgs>
void Grid<1, Real, Device, Index>::forBoundary(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 1, "Entity dimension must be either 0 or 1");

   auto outer = [=] __cuda_callable__(Index i, const Grid<1, Real, Device, Index>&grid, FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid);

      entity.setCoordinates(i);
      entity.refresh();

      func(entity, args...);
   };

   switch (EntityDimension) {
   case 0:
      // TODO: - Implement call within a single kernel
      TNL::Algorithms::ParallelFor<Device>::exec(0, 1, outer, *this, args...);
      TNL::Algorithms::ParallelFor<Device>::exec(this->getDimensions().x(), this->getDimensions().x() + 1, outer, *this, args...);
      break;
   case 1:
      TNL::Algorithms::ParallelFor<Device>::exec(0, 1, outer, *this, args...);
      TNL::Algorithms::ParallelFor<Device>::exec(this->getDimensions().x() - 1, this->getDimensions().x(), outer, *this, args...);

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
   default: break;
   }
}

template< typename Real,
          typename Device,
          typename Index >
template<int EntityDimension, typename Func, typename... FuncArgs>
void Grid<1, Real, Device, Index>::forInterior(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 1, "Entity dimension must be either 0 or 1");

   auto outer = [=] __cuda_callable__(Index i, const Grid<1, Real, Device, Index>&grid, FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid);

      entity.setCoordinates(i);
      entity.refresh();

      func(entity, args...);
   };

   switch (EntityDimension) {
   case 0:
      TNL::Algorithms::ParallelFor<Device>::exec(1, this->dimensions.x(), outer, *this, args...);
      break;
   case 1:
      TNL::Algorithms::ParallelFor<Device>::exec(1, this->dimensions.x() - 1, outer, *this, args...);

      // TODO: - Verify for distributed grids
      //TNL::Algorithms::ParallelFor<Device>::exec(interiorBegin.x(), interiorEnd.x(), outer, *this, args...);
      break;
   default: break;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real Grid< 1, Real, Device, Index >::
getSmallestSpaceStep() const
{
   return this->spaceSteps.x();
}

template< typename Real, typename Device, typename Index >
void
Grid< 1, Real, Device, Index >::writeProlog( Logger& logger ) const
{
   logger.writeParameter( "Dimension:", this->getMeshDimension() );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Space steps:", this->getSpaceSteps() );
   logger.writeParameter( "Number of cells:", getEntitiesCount< Cell >() );
   logger.writeParameter( "Number of vertices:", getEntitiesCount< Vertex >() );
}

}  // namespace Meshes
}  // namespace TNL
