// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Meshes {

template <class GridEntity, int NeighborEntityDimension>
class NeighborGridEntityGetter;

template<class>
class BoundaryGridEntityChecker;

template <class>
class GridEntityCenterGetter;

template <class, int>
class GridEntity {};

// TODO: - Implement
//       // compatibility with meshes, equivalent to getCenter
//       __cuda_callable__ inline
//       PointType getPoint() const;

//       __cuda_callable__ inline
//       const PointType& getEntityProportions() const;


template <int Dimension, typename Real, typename Device, typename Index, int EntityDimension>
class GridEntity<Meshes::Grid<Dimension, Real, Device, Index>, EntityDimension> {
   public:
      using IndexType = Index;
      using DeviceType = Device;
      using RealType = Real;

      using Grid = Meshes::Grid<Dimension, Real, Device, Index>;
      using Coordinate = typename Grid::Coordinate;
      using Point = typename Grid::Point;

      constexpr static int getMeshDimension() { return Grid::getMeshDimension(); };
      constexpr static int getEntityDimension() { return EntityDimension; };


   // template <int NeighborEntityDimension = getEntityDimension()>
   // using NeighborEntities =
   //     NeighborGridEntityGetter<GridEntity<Meshes::Grid<Dimension, Real, Device, Index>, EntityDimension, Config>, NeighborEntityDimension>;

      __cuda_callable__ inline
      GridEntity(const Grid& grid, const Coordinate& coordinates, const Coordinate& basis): grid(grid), coordinates(coordinates), basis(basis) {}

      __cuda_callable__ inline
      const Coordinate& getCoordinates() const;

      __cuda_callable__ inline
      Coordinate& getCoordinates();

      __cuda_callable__ inline
      void setCoordinates(const Coordinate& coordinates);

      /***
       * Call this method every time the coordinates are changed
       * to recompute the mesh entity index. The reason for this strange
       * mechanism is a performance.
       */
      __cuda_callable__ inline
      void refresh();

      __cuda_callable__ inline
      Index getIndex() const;

      __cuda_callable__ inline
      bool isBoundary() const;

      __cuda_callable__ inline
      const Point& getCenter() const;

      __cuda_callable__ inline
      Real getMeasure() const;

      __cuda_callable__ inline
      const Grid& getMesh() const;

      __cuda_callable__ inline
      void setBasis(const Coordinate& orientation);

      __cuda_callable__ inline
      const Coordinate& getBasis() const;
   protected:
      const Grid& grid;

      Index entityIndex;
      Coordinate coordinates;
      Coordinate basis;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/GridEntity_impl.h>
