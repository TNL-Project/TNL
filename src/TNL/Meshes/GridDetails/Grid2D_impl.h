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
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter2D_impl.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace TNL {
namespace Meshes {

template< typename Real, typename Device, typename Index >
Grid< 2, Real, Device, Index >::Grid( Index xSize, Index ySize )
{
   this->setDimensions( xSize, ySize );
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
void
Grid< 2, Real, Device, Index >::computeSpaceSteps()
{
   if( this->getDimensions().x() > 0 && this->getDimensions().y() > 0 ) {
      this->spaceSteps.x() = this->proportions.x() / (Real) this->getDimensions().x();
      this->spaceSteps.y() = this->proportions.y() / (Real) this->getDimensions().y();
      this->computeSpaceStepPowers();
   }
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
void
Grid< 2, Real, Device, Index >::computeSpaceStepPowers()
{
   const RealType& hx = this->spaceSteps.x();
   const RealType& hy = this->spaceSteps.y();

   Real auxX;
   Real auxY;
   for( int i = 0; i < 5; i++ ) {
      switch( i ) {
         case 0:
            auxX = 1.0 / ( hx * hx );
            break;
         case 1:
            auxX = 1.0 / hx;
            break;
         case 2:
            auxX = 1.0;
            break;
         case 3:
            auxX = hx;
            break;
         case 4:
            auxX = hx * hx;
            break;
      }
      for( int j = 0; j < 5; j++ ) {
         switch( j ) {
            case 0:
               auxY = 1.0 / ( hy * hy );
               break;
            case 1:
               auxY = 1.0 / hy;
               break;
            case 2:
               auxY = 1.0;
               break;
            case 3:
               auxY = hy;
               break;
            case 4:
               auxY = hy * hy;
               break;
         }
         this->spaceStepsProducts[ i ][ j ] = auxX * auxY;
      }
   }
}

template< typename Real, typename Device, typename Index >
void
Grid< 2, Real, Device, Index >::computeProportions()
{
   this->proportions.x() = this->dimensions.x() * this->spaceSteps.x();
   this->proportions.y() = this->dimensions.y() * this->spaceSteps.y();
}

template< typename Real, typename Device, typename Index >
void
Grid< 2, Real, Device, Index >::setDimensions( Index xSize, Index ySize )
{
   TNL_ASSERT_GE( xSize, 0, "Grid size must be non-negative." );
   TNL_ASSERT_GE( ySize, 0, "Grid size must be non-negative." );

   this->dimensions.x() = xSize;
   this->dimensions.y() = ySize;
   this->numberOfCells = xSize * ySize;
   this->numberOfNxFaces = ySize * ( xSize + 1 );
   this->numberOfNyFaces = xSize * ( ySize + 1 );
   this->numberOfFaces = this->numberOfNxFaces + this->numberOfNyFaces;
   this->numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );
   computeSpaceSteps();

   // only default behaviour, DistributedGrid must use the setters explicitly after setDimensions
   localBegin = 0;
   interiorBegin = 1;
   localEnd = dimensions;
   interiorEnd = dimensions - 1;
}

template< typename Real, typename Device, typename Index >
void
Grid< 2, Real, Device, Index >::setDimensions( const CoordinatesType& dimensions )
{
   return this->setDimensions( dimensions.x(), dimensions.y() );
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
inline const typename Grid< 2, Real, Device, Index >::CoordinatesType&
Grid< 2, Real, Device, Index >::getDimensions() const
{
   return this->dimensions;
}

template< typename Real, typename Device, typename Index >
void
Grid< 2, Real, Device, Index >::setLocalBegin( const CoordinatesType& begin )
{
   localBegin = begin;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< 2, Real, Device, Index >::CoordinatesType&
Grid< 2, Real, Device, Index >::getLocalBegin() const
{
   return localBegin;
}

template< typename Real, typename Device, typename Index >
void
Grid< 2, Real, Device, Index >::setLocalEnd( const CoordinatesType& end )
{
   localEnd = end;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< 2, Real, Device, Index >::CoordinatesType&
Grid< 2, Real, Device, Index >::getLocalEnd() const
{
   return localEnd;
}

template< typename Real, typename Device, typename Index >
void
Grid< 2, Real, Device, Index >::setInteriorBegin( const CoordinatesType& begin )
{
   interiorBegin = begin;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< 2, Real, Device, Index >::CoordinatesType&
Grid< 2, Real, Device, Index >::getInteriorBegin() const
{
   return interiorBegin;
}

template< typename Real, typename Device, typename Index >
void
Grid< 2, Real, Device, Index >::setInteriorEnd( const CoordinatesType& end )
{
   interiorEnd = end;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< 2, Real, Device, Index >::CoordinatesType&
Grid< 2, Real, Device, Index >::getInteriorEnd() const
{
   return interiorEnd;
}

template< typename Real, typename Device, typename Index >
void
Grid< 2, Real, Device, Index >::setDomain( const PointType& origin, const PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real, typename Device, typename Index >
void
Grid< 2, Real, Device, Index >::setOrigin( const PointType& origin )
{
   this->origin = origin;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
inline const typename Grid< 2, Real, Device, Index >::PointType&
Grid< 2, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
inline const typename Grid< 2, Real, Device, Index >::PointType&
Grid< 2, Real, Device, Index >::getProportions() const
{
   return this->proportions;
}

template< typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
inline Index
Grid< 2, Real, Device, Index >::getEntitiesCount() const
{
   static_assert( EntityDimension <= 2 && EntityDimension >= 0, "Wrong grid entity dimensions." );

   switch( EntityDimension ) {
      case 2:
         return this->numberOfCells;
      case 1:
         return this->numberOfFaces;
      case 0:
         return this->numberOfVertices;
   }
   return -1;
}

template< typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
inline Index
Grid< 2, Real, Device, Index >::getEntitiesCount() const
{
   return getEntitiesCount< Entity::getEntityDimension() >();
}

template< typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
inline Entity
Grid< 2, Real, Device, Index >::getEntity( const IndexType& entityIndex ) const
{
   static_assert( Entity::getEntityDimension() <= 2 && Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntity( *this, entityIndex );
}

template< typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
inline Index
Grid< 2, Real, Device, Index >::getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::getEntityDimension() <= 2 && Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntityIndex( *this, entity );
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
inline const typename Grid< 2, Real, Device, Index >::PointType&
Grid< 2, Real, Device, Index >::getSpaceSteps() const
{
   return this->spaceSteps;
}

template< typename Real, typename Device, typename Index >
inline void
Grid< 2, Real, Device, Index >::setSpaceSteps( const typename Grid< 2, Real, Device, Index >::PointType& steps )
{
   this->spaceSteps = steps;
   computeSpaceStepPowers();
   computeProportions();
}

template< typename Real, typename Device, typename Index >
template< int xPow, int yPow >
__cuda_callable__
inline const Real&
Grid< 2, Real, Device, Index >::getSpaceStepsProducts() const
{
   static_assert( xPow >= -2 && xPow <= 2, "unsupported value of xPow" );
   static_assert( yPow >= -2 && yPow <= 2, "unsupported value of yPow" );
   return this->spaceStepsProducts[ xPow + 2 ][ yPow + 2 ];
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
Index
Grid< 2, Real, Device, Index >::getNumberOfNxFaces() const
{
   return numberOfNxFaces;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
const Real&
Grid< 2, Real, Device, Index >::getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1, 1 >();
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
inline Real
Grid< 2, Real, Device, Index >::getSmallestSpaceStep() const
{
   return min( this->spaceSteps.x(), this->spaceSteps.y() );
}

template< typename Real,
          typename Device,
          typename Index >
template<int EntityDimension, typename Func, typename... FuncArgs>
void Grid<2, Real, Device, Index>::forAll(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 2, "Entity dimension must be in range [0..<2]");

   auto outer = [=] __cuda_callable__(Index i, Index j, const Grid<2, Real, Device, Index>&grid, FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid, CoordinatesType(i, j));
      entity.refresh();

      func(entity, args...);
   };

   switch (EntityDimension) {
   case 0:
      TNL::Algorithms::ParallelFor2D<Device>::exec(0, 0, dimensions.x() + 1, dimensions.y() + 1, outer, *this, args...);
      break;
   case 1: {
      auto outerOriented = [=] __cuda_callable__(Index i, Index j,
                                                 const Grid<2, Real, Device, Index>&grid,
                                                 const CoordinatesType & orientation,
                                                 FuncArgs... args) mutable {
         EntityType<EntityDimension> entity(grid, CoordinatesType(i, j), orientation);

         entity.refresh();

         func(entity, args...);
      };

      TNL::Algorithms::ParallelFor2D<Device>::exec(0, 0,
                                                   dimensions.x() + 1, dimensions.y(),
                                                   outerOriented,
                                                   *this,
                                                   CoordinatesType(1, 0),
                                                   args...);

      TNL::Algorithms::ParallelFor2D<Device>::exec(0, 0,
                                                   dimensions.x(), dimensions.y() + 1,
                                                   outerOriented,
                                                   *this,
                                                   CoordinatesType(0, 1),
                                                   args...);
      break;
   }
   case 2:
      TNL::Algorithms::ParallelFor2D<Device>::exec(0, 0, dimensions.x(), dimensions.y(), outer, *this, args...);

      // TODO: - Verify for distributed grids
      //TNL::Algorithms::ParallelFor2D<Device>::exec(localBegin.x(), localBegin.y(), localEnd.x(), localEnd.y(), outer, * this, args...);
      break;
   default: break;
   }
}

template< typename Real,
          typename Device,
          typename Index >
template<int EntityDimension, typename Func, typename... FuncArgs>
void Grid<2, Real, Device, Index>::forInterior(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 2, "Entity dimension must be in range [0..<2]");

   auto outer = [=] __cuda_callable__(Index i, Index j, const Grid<2, Real, Device, Index>&grid, FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid);

      entity.setCoordinates({ i, j });
      entity.refresh();

      func(entity, args...);
   };

   switch (EntityDimension) {
   case 0:
      TNL::Algorithms::ParallelFor2D<Device>::exec(1, 1, dimensions.x(), dimensions.y(), outer, *this, args...);
      break;
   case 1: {
      auto outerOriented = [=] __cuda_callable__(Index i, Index j,
                                                 Grid<2, Real, Device, Index>& grid,
                                                 const CoordinatesType& orientation,
                                                 FuncArgs... args) mutable {
         EntityType<EntityDimension> entity(grid, CoordinatesType(i, j), orientation);

         entity.refresh();

         func(entity, args...);
      };

      TNL::Algorithms::ParallelFor2D<Device>::exec(1, 0,
                                                   dimensions.x(), dimensions.y(),
                                                   outerOriented,
                                                   *this,
                                                   CoordinatesType(1, 0),
                                                   args...);

      TNL::Algorithms::ParallelFor2D<Device>::exec(0, 1,
                                                   dimensions.x(), dimensions.y(),
                                                   outerOriented,
                                                   *this,
                                                   CoordinatesType(0, 1),
                                                   args...);
      break;
   }
   case 2:
      TNL::Algorithms::ParallelFor2D<Device>::exec(1, 1, dimensions.x() - 1, dimensions.y() - 1, outer, *this, args...);

      // TODO: - Verify for distributed grids
      //TNL::Algorithms::ParallelFor2D<Device>::exec(interiorBegin.x(), interiorBegin.y(), interiorEnd.x(), interiorEnd.y(), outer, *this, args...);
      break;
   default:
      break;
   }
}

template< typename Real,
          typename Device,
          typename Index >
template<int EntityDimension, typename Func, typename... FuncArgs>
void Grid<2, Real, Device, Index>::forBoundary(Func func, FuncArgs... args) const {
   static_assert(EntityDimension >= 0 && EntityDimension <= 2, "Entity dimension must be in range [0...2]");

   auto outer = [=] __cuda_callable__(Index i, Index axis, Index axisIndex, const Grid<2, Real, Device, Index>&grid, FuncArgs... args) mutable {
      EntityType<EntityDimension> entity(grid);

      switch (axis) {
      case 0:
         entity.setCoordinates({ axisIndex, i });
         break;
      case 1:
         entity.setCoordinates({ i, axisIndex });
         break;
      default: TNL_ASSERT_TRUE(false, "Received axis index. Expect in range [0..<1]");
      }

      entity.refresh();

      func(entity, args...);
   };

   switch (EntityDimension) {
   case 0:
      // Lower horizontal
      TNL::Algorithms::ParallelFor<Device>::exec(0, dimensions.x() + 1, outer, 0, 0, *this, args...);
      // Upper horizontal
      TNL::Algorithms::ParallelFor<Device>::exec(0, dimensions.x() + 1, outer, 0, dimensions.y(), *this, args...);
      // Left vertical
      TNL::Algorithms::ParallelFor<Device>::exec(0, dimensions.y() + 1, outer, 1, 0, *this, args...);
      // Right vertical
      TNL::Algorithms::ParallelFor<Device>::exec(0, dimensions.y() + 1, outer, 1, dimensions.x(), *this, args...);
      break;
   case 1: {
      auto outerOriented = [=] __cuda_callable__(Index i,
                                                 Index axis,
                                                 Index axisIndex,
                                                 const Grid<2, Real, Device, Index>& grid,
                                                 const CoordinatesType& orientation,
                                                 FuncArgs... args) mutable {
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
         default: TNL_ASSERT_TRUE(false, "Received axis index. Expect in range [0...1]");
         }

         EntityType<EntityDimension> entity(grid, coordinates, orientation);

         entity.refresh();

         func(entity, args...);
      };

      // Lower horizontal
      TNL::Algorithms::ParallelFor<Device>::exec(0,
                                                 dimensions.x(),
                                                 outerOriented,
                                                 0, 0,
                                                 *this,
                                                 CoordinatesType(1, 0),
                                                 args...);
      // Upper horizontal
      TNL::Algorithms::ParallelFor<Device>::exec(0,
                                                 dimensions.x(),
                                                 outerOriented,
                                                 0, dimensions.y(),
                                                 *this,
                                                 CoordinatesType(1, 0),
                                                 args...);
      // Left vertical
      TNL::Algorithms::ParallelFor<Device>::exec(0,
                                                 dimensions.y(),
                                                 outerOriented,
                                                 1, 0,
                                                 *this,
                                                 CoordinatesType(0, 1),
                                                 args...);
      // Right vertical
      TNL::Algorithms::ParallelFor<Device>::exec(0,
                                                 dimensions.y(),
                                                 outerOriented,
                                                 1, dimensions.x(),
                                                 *this,
                                                 CoordinatesType(0, 1),
                                                 args...);
      break;
   }
   case 2:
      // Lower horizontal
      TNL::Algorithms::ParallelFor<Device>::exec(0, dimensions.x(), outer, 0, 0, *this, args...);
      // Upper horizontal
      TNL::Algorithms::ParallelFor<Device>::exec(0, dimensions.x(), outer, 0, dimensions.y() - 1, *this, args...);
      // Left vertical
      TNL::Algorithms::ParallelFor<Device>::exec(0, dimensions.y(), outer, 1, 0, *this, args...);
      // Right vertical
      TNL::Algorithms::ParallelFor<Device>::exec(0, dimensions.y(), outer, 1, dimensions.x() - 1, *this, args...);
      // TODO: - Verify with the distributed grid
      /*if (localBegin < interiorBegin && interiorEnd < localEnd) {
         // Lower horizontal
         TNL::Algorithms::ParallelFor2D<Device>::exec(interiorBegin.x() - 1, interiorBegin.y() - 1, interiorEnd.x() + 1, interiorBegin.y() + 1, outer, *this, args...);
         // Upper horizontal
         TNL::Algorithms::ParallelFor2D<Device>::exec(interiorBegin.x() - 1, interiorEnd.y() - 1, interiorEnd.x() + 1, interiorEnd.y() + 1, outer, *this, args...);
         // Left vertical
         TNL::Algorithms::ParallelFor2D<Device>::exec(interiorBegin.x() -1, interiorBegin.y() - 1, interiorBegin.x() + 1, interiorEnd.y() + 1, outer, *this, args...);
         // Right vertical
         TNL::Algorithms::ParallelFor2D<Device>::exec(interiorEnd.x() - 1, interiorBegin.y() - 1, interiorEnd.x() + 1, interiorEnd.y() + 1, outer, *this, args...);
         return;
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

template< typename Real,
          typename Device,
          typename Index >
void
Grid< 2, Real, Device, Index >::writeProlog( Logger& logger ) const
{
   logger.writeParameter( "Dimension:", getMeshDimension() );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Space steps:", this->getSpaceSteps() );
   logger.writeParameter( "Number of cells:", getEntitiesCount< Cell >() );
   logger.writeParameter( "Number of faces:", getEntitiesCount< Face >() );
   logger.writeParameter( "Number of vertices:", getEntitiesCount< Vertex >() );
}

}  // namespace Meshes
}  // namespace TNL
