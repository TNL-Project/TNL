// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace TNL {
namespace Meshes {

#define __GRID_1D_TEMPLATE__ template< typename Real, typename Device, typename Index >
#define __GRID_1D_PREFIX__ Grid< 1, Real, Device, Index >

__GRID_1D_TEMPLATE__
__GRID_1D_PREFIX__::Grid( const Index xSize )
{
   this->setDimensions( xSize );
}

__GRID_1D_TEMPLATE__
template< typename Entity >
__cuda_callable__
inline Index
__GRID_1D_PREFIX__::getEntityIndex( const Entity& entity ) const
{
   static_assert( Templates::isInClosedInterval( 0, Entity::entityDimension, 1 ), "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity::entityDimension >::getEntityIndex( *this, entity );
}

__GRID_1D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_1D_PREFIX__::forAll( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );
      entity.refresh();

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( exec, *this, args... );
}

__GRID_1D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_1D_PREFIX__::forAll( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );
      entity.refresh();

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( from, to, exec, *this, args... );
}

__GRID_1D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_1D_PREFIX__::forBoundary( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      entity.refresh();

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( exec, *this, args... );
}

__GRID_1D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_1D_PREFIX__::forBoundary( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      entity.refresh();

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( from, to, exec, *this, args... );
}

__GRID_1D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_1D_PREFIX__::forInterior( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      entity.refresh();

      func( entity, args... );
   };

   this->template traverseInterior< EntityDimension >( exec, *this, args... );
}

__GRID_1D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_1D_PREFIX__::forInterior( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      entity.refresh();

      func( entity, args... );
   };

   this->template traverseInterior< EntityDimension >( from, to, exec, *this, args... );
}

}  // namespace Meshes
}  // namespace TNL
