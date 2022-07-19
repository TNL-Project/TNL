// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Logger.h>
#include <TNL/Meshes/GridDetails/NDGrid.h>

namespace TNL {
namespace Meshes {

template< class, int >
class GridEntity;

template< int Dimension, typename Real = double, typename Device = Devices::Host, typename Index = int >
class Grid : public NDGrid< Dimension, Real, Device, Index >
{
public:

   template< int EntityDimension >
   using EntityType = GridEntity< Grid, EntityDimension >;

   using Base = NDGrid< Dimension, Real, Device, Index >;
   using Coordinate = typename Base::Coordinate;
   using Point = typename Base::Point;
   using EntitiesCounts = typename Base::EntitiesCounts;


   /**
    * @brief Traverser all elements in rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forAll( Func func, FuncArgs... args ) const;

   /**
    * @brief Traverser all elements in rect
    * @param from - bottom left anchor of traverse rect
    * @param to - top right anchor of traverse rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forAll( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const;

   /**
    * @brief Traverser interior elements in rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forInterior( Func func, FuncArgs... args ) const;

   /**
    * @brief Traverser interior elements
    * @param from - bottom left anchor of traverse rect
    * @param to - top right anchor of traverse rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forInterior( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const;

   /**
    * @brief Traverser boundary elements in rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forBoundary( Func func, FuncArgs... args ) const;

   /**
    * @brief Traverser boundary elements in rect
    * @param from - bottom left anchor of traverse rect
    * @param to - top right anchor of traverse rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forBoundary( const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args ) const;

};

template< int Dimension, typename Real, typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str, const Grid< Dimension, Real, Device, Index >& grid )
{
   TNL::Logger logger( 100, str );

   grid.writeProlog( logger );

   return str;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
operator==( const Grid< Dimension, Real, Device, Index >& lhs, const Grid< Dimension, Real, Device, Index >& rhs )
{
   return lhs.getDimensions() == rhs.getDimensions() && lhs.getOrigin() == rhs.getOrigin()
       && lhs.getProportions() == rhs.getProportions();
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
operator!=( const Grid< Dimension, Real, Device, Index >& lhs, const Grid< Dimension, Real, Device, Index >& rhs )
{
   return ! ( lhs == rhs );
}

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>
#include <TNL/Meshes/GridDetails/Grid.hpp>
