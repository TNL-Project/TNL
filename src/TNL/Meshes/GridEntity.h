// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntitiesStorage.h>

namespace TNL {
namespace Meshes {

template <typename GridEntity, int NeighborEntityDimension, typename StencilStorage>
class NeighborGridEntityGetter;

template <typename GridEntityType>
class BoundaryGridEntityChecker;

template <typename GridEntityType>
class GridEntityCenterGetter;

template <typename Grid, int EntityDimension>
class GridEntity;


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

      using Coordinate = typename Grid::Coordinate;
      using Point = typename Grid::Point;
      using Grid = Meshes::Grid<Dimension, Real, Device, Index>;

      constexpr static int getMeshDimension() { return GridType::getMeshDimension(); };
      constexpr static int getEntityDimension() { return EntityDimension; };


   // template <int NeighborEntityDimension = getEntityDimension()>
   // using NeighborEntities =
   //     NeighborGridEntityGetter<GridEntity<Meshes::Grid<Dimension, Real, Device, Index>, EntityDimension, Config>, NeighborEntityDimension>;

      __cuda_callable__ inline
      GridEntity(const GridType& grid, const Coordinate& coordinates, const Coordinate& basis);


      __cuda_callable__ inline
      const Coordinate& getCoordinates() const;

      __cuda_callable__ inline
      Coordinate& getCoordinates() const;

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
