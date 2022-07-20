// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/NDGrid.h>

namespace TNL {
namespace Meshes {

template< class, int >
class GridEntity;

template< typename Real, typename Device, typename Index >
class Grid< 3, Real, Device, Index > : public NDGrid< 3, Real, Device, Index >
{
public:
   template< int EntityDimension >
   using EntityType = GridEntity< Grid, EntityDimension >;

   using Base = NDGrid< 3, Real, Device, Index >;
   using CoordinatesType = typename Base::CoordinatesType;
   using PointType = typename Base::PointType;
   using EntitiesCounts = typename Base::EntitiesCounts;

   /**
    * \brief Returns the dimension of grid
    */
   static constexpr int
   getMeshDimension()
   {
      return 3;
   };

   Grid() = default;
   Grid( const Index xSize, const Index ySize, const Index zSize );

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
   forAll( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const;

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
   forInterior( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const;

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
   forBoundary( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/Grid3D.hpp>
