/***************************************************************************
                          Grid3D_impl.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/GridDetails/Grid3D.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace TNL {
namespace Meshes {

#define __GRID_3D_TEMPLATE__ template< typename Real, typename Device, typename Index >
#define __GRID_3D_PREFIX__ Grid< 3, Real, Device, Index >

__GRID_3D_TEMPLATE__
__GRID_3D_PREFIX__::Grid( const Index xSize, const Index ySize, const Index zSize )
{
   this->setDimensions( xSize, ySize, zSize );
}

__GRID_3D_TEMPLATE__
template< typename Entity >
__cuda_callable__
inline Index
__GRID_3D_PREFIX__::getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::entityDimension <= 3 && Entity::entityDimension >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity::entityDimension >::getEntityIndex( *this, entity );
}

__GRID_3D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_3D_PREFIX__::forAll( Func func, FuncArgs... args ) const
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

__GRID_3D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_3D_PREFIX__::forAll( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
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

__GRID_3D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_3D_PREFIX__::forInterior( Func func, FuncArgs... args ) const
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

__GRID_3D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_3D_PREFIX__::forInterior( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
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

__GRID_3D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_3D_PREFIX__::forBoundary( Func func, FuncArgs... args ) const
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

__GRID_3D_TEMPLATE__
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
__GRID_3D_PREFIX__::forBoundary( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
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

}  // namespace Meshes
}  // namespace TNL
