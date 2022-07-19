// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace TNL {
namespace Meshes {

template< typename Real, typename Device, typename Index >
Grid< 2, Real, Device, Index >::Grid( const Index xSize, const Index ySize )
{
   this->setDimensions( xSize, ySize );
}

/*template< typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
inline Index
Grid< 2, Real, Device, Index >::getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::entityDimension <= 2 && Entity::entityDimension >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity::entityDimension >::getEntityIndex( *this, entity );
}*/

template< typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
Grid< 2, Real, Device, Index >::forAll( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( exec, *this, args... );
}

template< typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
Grid< 2, Real, Device, Index >::forAll( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( from, to, exec, *this, args... );
}

template< typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
Grid< 2, Real, Device, Index >::forInterior( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      func( entity, args... );
   };

   this->template traverseInterior< EntityDimension >( exec, *this, args... );
}

template< typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
Grid< 2, Real, Device, Index >::forInterior( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      func( entity, args... );
   };

   this->template traverseInterior< EntityDimension >( from, to, exec, *this, args... );
}

template< typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
Grid< 2, Real, Device, Index >::forBoundary( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( exec, *this, args... );
}

template< typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
Grid< 2, Real, Device, Index >::forBoundary( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const Coordinate& coordinate,
                                        const Coordinate& basis,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, basis, orientation );

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( from, to, exec, *this, args... );
}

}  // namespace Meshes
}  // namespace TNL
