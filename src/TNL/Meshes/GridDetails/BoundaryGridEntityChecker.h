// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL {
namespace Meshes {

template<class>
class BoundaryGridEntityChecker;

/***
 * 1D grids
 */
template< typename Real,
          typename Device,
          typename Index >
class BoundaryGridEntityChecker<GridEntity<Meshes::Grid<1, Real, Device, Index>, 1>>
{
  public:
      using Entity = GridEntity<Meshes::Grid<1, Real, Device, Index>, 1>;

      __cuda_callable__ inline
      static bool isBoundaryEntity(const Entity& entity) {
         return (entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1);
      }
};

template< typename Real,
          typename Device,
          typename Index>
class BoundaryGridEntityChecker<GridEntity<Meshes::Grid<1, Real, Device, Index>, 0>>
{
   public:
       using Entity = GridEntity<Meshes::Grid<1, Real, Device, Index>, 0>;

      __cuda_callable__ inline
      static bool isBoundaryEntity(const Entity& entity) {
         return (entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x());
      }
};

/****
 * 2D grids
 */
template<typename Real,
         typename Device,
         typename Index >
class BoundaryGridEntityChecker<GridEntity< Meshes::Grid<2, Real, Device, Index>, 2>>
{
   public:
      using Entity = GridEntity<Meshes::Grid<2, Real, Device, Index>, 2>;

      __cuda_callable__ inline
      static bool isBoundaryEntity(const Entity& entity)
      {
         return (entity.getCoordinates().x() == 0 ||
                 entity.getCoordinates().y() == 0 ||
                 entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 ||
                 entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1);
      }
};

template<typename Real,
         typename Device,
         typename Index>
class BoundaryGridEntityChecker<GridEntity< Meshes::Grid<2, Real, Device, Index>, 1>>
{
   public:
      using Entity = GridEntity<Meshes::Grid<2, Real, Device, Index>, 1>;

      __cuda_callable__ inline
      static bool isBoundaryEntity(const Entity& entity)
      {
         return ((entity.getOrientation().x() &&
                  (entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x())) ||
                 (entity.getOrientation().y() &&
                  (entity.getCoordinates().y() == 0 || entity.getCoordinates().y() == entity.getMesh().getDimensions().y())));
      }
};

template <typename Real,
          typename Device,
          typename Index>
class BoundaryGridEntityChecker<GridEntity<Meshes::Grid<2, Real, Device, Index>, 0>> {
   public:
      using Entity = GridEntity<Meshes::Grid<2, Real, Device, Index>, 0>;

      __cuda_callable__ inline
      static bool isBoundaryEntity(const Entity& entity) {
         return (entity.getCoordinates().x() == 0 || entity.getCoordinates().y() == 0 ||
                 entity.getCoordinates().x() == entity.getMesh().getDimensions().x() ||
                 entity.getCoordinates().y() == entity.getMesh().getDimensions().y());
      }
};

/***
 * 3D grid
 */
template<typename Real,
         typename Device,
         typename Index>
class BoundaryGridEntityChecker<GridEntity< Meshes::Grid<3, Real, Device, Index>, 3>>
{
   public:
      using Entity = GridEntity<Meshes::Grid<3, Real, Device, Index>, 3>;

      __cuda_callable__ inline
      static bool isBoundaryEntity(const Entity& entity) {
         return (entity.getCoordinates().x() == 0 || entity.getCoordinates().y() == 0 || entity.getCoordinates().z() == 0 ||
                 entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 ||
                 entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 ||
                 entity.getCoordinates().z() == entity.getMesh().getDimensions().z() - 1);
      }
};

template<typename Real,
         typename Device,
         typename Index>
class BoundaryGridEntityChecker<GridEntity<Meshes::Grid<3, Real, Device, Index>, 2>>
{
   public:
      using Entity = GridEntity<Meshes::Grid<3, Real, Device, Index>, 2>;

      __cuda_callable__ inline
      static bool isBoundaryEntity(const Entity& entity) {
         return ((entity.getOrientation().x() &&
                  (entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x())) ||
                 (entity.getOrientation().y() &&
                  (entity.getCoordinates().y() == 0 || entity.getCoordinates().y() == entity.getMesh().getDimensions().y())) ||
                 (entity.getOrientation().z() &&
                  (entity.getCoordinates().z() == 0 || entity.getCoordinates().z() == entity.getMesh().getDimensions().z())));
      }
};

template<typename Real,
         typename Device,
         typename Index>
class BoundaryGridEntityChecker<GridEntity<Meshes::Grid<3, Real, Device, Index>, 1>> {
  public:
      using Entity = GridEntity<Meshes::Grid<3, Real, Device, Index>, 1>;

      __cuda_callable__ inline static bool isBoundaryEntity(const Entity& entity) {
         return (
            (entity.getOrientation().x() &&
            (entity.getCoordinates().y() == 0 || entity.getCoordinates().y() == entity.getMesh().getDimensions().y() ||
               entity.getCoordinates().z() == 0 || entity.getCoordinates().z() == entity.getMesh().getDimensions().z())) ||
            (entity.getOrientation().y() &&
            (entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() ||
               entity.getCoordinates().z() == 0 || entity.getCoordinates().z() == entity.getMesh().getDimensions().z())) ||
            (entity.getOrientation().z() && (entity.getCoordinates().x() == 0 || entity.getCoordinates().x() == entity.getMesh().getDimensions().x() ||
                                             entity.getCoordinates().y() == 0 || entity.getCoordinates().y() == entity.getMesh().getDimensions().y())));
      }
};

template <typename Real,
          typename Device,
          typename Index>
class BoundaryGridEntityChecker<GridEntity<Meshes::Grid<3, Real, Device, Index>, 0>> {
  public:
   using Entity = GridEntity<Meshes::Grid<3, Real, Device, Index>, 1>;

   __cuda_callable__ inline static bool isBoundaryEntity(const Entity& entity) {
      return (entity.getCoordinates().x() == 0 || entity.getCoordinates().y() == 0 || entity.getCoordinates().z() == 0 ||
              entity.getCoordinates().x() == entity.getMesh().getDimensions().x() ||
              entity.getCoordinates().y() == entity.getMesh().getDimensions().y() ||
              entity.getCoordinates().z() == entity.getMesh().getDimensions().z());
   }
};

} // namespace Meshes
} // namespace TNL
