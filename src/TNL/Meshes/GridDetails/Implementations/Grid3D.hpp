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

template< typename Real, typename Device, typename Index >
Grid< 3, Real, Device, Index >::Grid( const Index xSize, const Index ySize, const Index zSize )
{
   this->setDimensions( xSize, ySize, zSize );
}

template< typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
inline void
Grid< 3, Real, Device, Index >::forAll( Func func, FuncArgs... args ) const
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
Grid< 3, Real, Device, Index >::forAll( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
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
Grid< 3, Real, Device, Index >::forInterior( Func func, FuncArgs... args ) const
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
Grid< 3, Real, Device, Index >::forInterior( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
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
Grid< 3, Real, Device, Index >::forBoundary( Func func, FuncArgs... args ) const
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
Grid< 3, Real, Device, Index >::forBoundary( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const
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
