/***************************************************************************
                          BoundaryGridEntityChecker.h  -  description
                             -------------------
    begin                : Dec 2, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/GridDetails/BoundaryGridEntityChecker.h>

namespace TNL {
namespace Meshes {

template< int Dimension, typename Real, typename Device, typename Index >
class Grid;

/***
 * 1D grids
 */
template< typename Real, typename Device, typename Index >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1 > >
{
public:
   using Entity = GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1 >;

   __cuda_callable__
   inline static bool
   isBoundaryEntity( const Entity& entity )
   {
      return ( entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 );
   }
};

template< typename Real, typename Device, typename Index >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0 > >
{
public:
   using Entity = GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0 >;

   __cuda_callable__
   inline static bool
   isBoundaryEntity( const Entity& entity )
   {
      return ( entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() );
   }
};

/****
 * 2D grids
 */
template< typename Real, typename Device, typename Index >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2 > >
{
public:
   using Entity = GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2 >;

   __cuda_callable__
   inline static bool
   isBoundaryEntity( const Entity& entity )
   {
      return ( entity.getCoordinates().x() == 0 || entity.getCoordinates().y() == 0
               || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1
               || entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 );
   }
};

template< typename Real, typename Device, typename Index >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 1 > >
{
public:
   using Entity = GridEntity< Meshes::Grid< 2, Real, Device, Index >, 1 >;

   __cuda_callable__
   inline static bool
   isBoundaryEntity( const Entity& entity )
   {
      return (
         ( entity.getBasis().x()
           && ( entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() ) )
         || ( entity.getBasis().y()
              && ( entity.getCoordinates().y() == 0
                   || entity.getCoordinates().y() == entity.getMesh().getDimensions().y() ) ) );
   }
};

template< typename Real, typename Device, typename Index >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 0 > >
{
public:
   using Entity = GridEntity< Meshes::Grid< 2, Real, Device, Index >, 0 >;

   __cuda_callable__
   inline static bool
   isBoundaryEntity( const Entity& entity )
   {
      return ( entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() )
          || ( entity.getCoordinates().y() == 0 || entity.getCoordinates().y() == entity.getMesh().getDimensions().y() );
   }
};

/***
 * 3D grid
 */
template< typename Real, typename Device, typename Index >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3 > >
{
public:
   using Entity = GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3 >;

   __cuda_callable__
   inline static bool
   isBoundaryEntity( const Entity& entity )
   {
      return ( entity.getCoordinates().x() == 0 || entity.getCoordinates().y() == 0 || entity.getCoordinates().z() == 0
               || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1
               || entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1
               || entity.getCoordinates().z() == entity.getMesh().getDimensions().z() - 1 );
   }
};

template< typename Real, typename Device, typename Index >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 2 > >
{
public:
   using Entity = GridEntity< Meshes::Grid< 3, Real, Device, Index >, 2 >;

   __cuda_callable__
   inline static bool
   isBoundaryEntity( const Entity& entity )
   {
      return (
         ( entity.getBasis().x()
           && ( entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() ) )
         || ( entity.getBasis().y()
              && ( entity.getCoordinates().y() == 0 || entity.getCoordinates().y() == entity.getMesh().getDimensions().y() ) )
         || ( entity.getBasis().z()
              && ( entity.getCoordinates().z() == 0
                   || entity.getCoordinates().z() == entity.getMesh().getDimensions().z() ) ) );
   }
};

template< typename Real, typename Device, typename Index >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 1 > >
{
public:
   using Entity = GridEntity< Meshes::Grid< 3, Real, Device, Index >, 1 >;

   __cuda_callable__
   inline static bool
   isBoundaryEntity( const Entity& entity )
   {
      return (
         ( entity.getBasis().x()
           && ( entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() ) )
         || ( entity.getBasis().y()
              && ( entity.getCoordinates().y() == 0 || entity.getCoordinates().y() == entity.getMesh().getDimensions().y() ) )
         || ( entity.getBasis().z()
              && ( entity.getCoordinates().z() == 0
                   || entity.getCoordinates().z() == entity.getMesh().getDimensions().z() ) ) );
   }
};

template< typename Real, typename Device, typename Index >
class BoundaryGridEntityChecker< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 0 > >
{
public:
   using Entity = GridEntity< Meshes::Grid< 3, Real, Device, Index >, 0 >;

   __cuda_callable__
   inline static bool
   isBoundaryEntity( const Entity& entity )
   {
      return ( entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() )
          || ( entity.getCoordinates().y() == 0 || entity.getCoordinates().y() == entity.getMesh().getDimensions().y() )
          || ( entity.getCoordinates().z() == 0 || entity.getCoordinates().z() == entity.getMesh().getDimensions().z() );
   }
};

}  // namespace Meshes
}  // namespace TNL
