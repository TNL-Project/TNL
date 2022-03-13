// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridEntity.h>

namespace TNL {
namespace Meshes {

template<typename, int>
class GridEntityGetter;

/****
 * 1D grid
 */
template <typename Real, typename Device, typename Index, int EntityDimension>
class GridEntityGetter<Meshes::Grid<1, Real, Device, Index>, EntityDimension> {
   public:
      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      static Entity getEntity(const Grid& grid, const Index& index) {
         TNL_ASSERT_GE(index, 0, "Index must be non-negative.");
         TNL_ASSERT_LT(index, grid.template getEntitiesCount<Entity>(), "Index is out of bounds.");
         return Entity(grid, Coordinate(index), Coordinate(0));
      }

      __cuda_callable__ inline
      static Index getEntityIndex(const Grid& grid, const Entity& entity) {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), grid.getDimensions() + Coordinate(1 - EntityDimension), "wrong coordinates");

         return entity.getCoordinates().x();
      }
};

/****
 * 2D grid
 */
template <typename Real, typename Device, typename Index>
class GridEntityGetter<Meshes::Grid<2, Real, Device, Index>, 2> {
   public:
      static constexpr int EntityDimension = 2;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      static Entity getEntity(const Grid& grid, const Index& index) {
         TNL_ASSERT_GE(index, 0, "Index must be non-negative.");
         TNL_ASSERT_LT(index, grid.template getEntitiesCount<Entity>(), "Index is out of bounds.");

         const Coordinate dimensions = grid.getDimensions();

         return Entity(grid, Coordinate(index % dimensions.x(), index / dimensions.x()), Coordinate(0, 0));
      }

      __cuda_callable__ inline
      static Index getEntityIndex(const Grid& grid, const Entity& entity) {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), grid.getDimensions(), "wrong coordinates");

         return entity.getCoordinates().y() * grid.getDimensions().x() + entity.getCoordinates().x();
      }
};

template <typename Real, typename Device, typename Index>
class GridEntityGetter<Meshes::Grid<2, Real, Device, Index>, 1> {
   public:
      static constexpr int EntityDimension = 1;

      using Grid = Meshes::Grid<2, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      static Entity getEntity(const Grid& grid, const Index& index) {
         TNL_ASSERT_GE(index, 0, "Index must be non-negative.");
         TNL_ASSERT_LT(index, grid.template getEntitiesCount<Entity>(), "Index is out of bounds.");

         const Coordinate dimensions = grid.getDimensions();

         if (index < grid.numberOfNxFaces) {
            const Index aux = dimensions.x() + 1;
            return Entity(grid, Coordinate(index % aux, index / aux), Coordinate(1, 0));
         }
         const Index i = index - grid.numberOfNxFaces;
         const Index& aux = dimensions.x();
         return Entity(grid, Coordinate(i % aux, i / aux), Coordinate(0, 1));
      }

      __cuda_callable__ inline
      static Index getEntityIndex(const Grid& grid, const Entity& entity) {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
         // TNL_ASSERT_LT(entity.getCoordinates(), grid.getDimensions() + abs(entity.getOrientation()), "wrong coordinates");

         const Coordinate& coordinates = entity.getCoordinates();
         const Coordinate& dimensions = grid.getDimensions();

         // if (entity.getOrientation().x()) return coordinates.y() * (dimensions.x() + 1) + coordinates.x();

         return grid.template getOrientedEntitiesCount<2, 0>() + coordinates.y() * dimensions.x() + coordinates.x();
      }
};

template <typename Real, typename Device, typename Index>
class GridEntityGetter<Meshes::Grid<2, Real, Device, Index>, 0> {
   public:
      static constexpr int EntityDimension = 0;

      using Grid = Meshes::Grid<2, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      static Entity getEntity(const Grid& grid, const Index& index) {
         TNL_ASSERT_GE(index, 0, "Index must be non-negative.");
         TNL_ASSERT_LT(index, grid.template getEntitiesCount<Entity>(), "Index is out of bounds.");

         const Coordinate dimensions = grid.getDimensions();

         const Index aux = dimensions.x() + 1;
         return Entity(grid, Coordinate(index % aux, index / aux), Coordinate(0, 0));
      }

      __cuda_callable__ inline
      static Index getEntityIndex(const Grid& grid, const Entity& entity) {
         // TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
         // TNL_ASSERT_LE(entity.getCoordinates(), grid.getDimensions(), "wrong coordinates");

         const Coordinate& coordinates = entity.getCoordinates();
         const Coordinate& dimensions = grid.getDimensions();

         return coordinates.y() * (dimensions.x() + 1) + coordinates.x();
      }
};

/****
 * 3D grid
 */
template <typename Real, typename Device, typename Index>
class GridEntityGetter<Meshes::Grid<3, Real, Device, Index>, 3> {
   public:
      static constexpr int EntityDimension = 3;

      using Grid = Meshes::Grid<3, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      static Entity getEntity(const Grid& grid, const Index& index) {
         TNL_ASSERT_GE(index, 0, "Index must be non-negative.");
         TNL_ASSERT_LT(index, grid.template getEntitiesCount<Entity>(), "Index is out of bounds.");

         const Coordinate dimensions = grid.getDimensions();

         return Entity(grid,
                           Coordinate(index % dimensions.x(), (index / dimensions.x()) % dimensions.y(), index / (dimensions.x() * dimensions.y())),
                           Coordinate(0, 0, 0));
      }

      __cuda_callable__ inline
      static Index getEntityIndex(const Grid& grid, const Entity& entity) {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), grid.getDimensions(), "wrong coordinates");

         const Coordinate coordinates = entity.getCoordinates();
         const Coordinate dimensions = grid.getDimensions();

         return (coordinates.z() * dimensions.y() + coordinates.y()) * dimensions.x() + coordinates.x();
      }
};

template <typename Real, typename Device, typename Index>
class GridEntityGetter<Meshes::Grid<3, Real, Device, Index>, 2> {
   public:
      static constexpr int EntityDimension = 2;

      using Grid = Meshes::Grid<3, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      static Entity getEntity(const Grid& grid, const Index& index) {
         TNL_ASSERT_GE(index, 0, "Index must be non-negative.");
         TNL_ASSERT_LT(index, grid.template getEntitiesCount<Entity>(), "Index is out of bounds.");

         const Coordinate dimensions = grid.getDimensions();

         if (index < grid.numberOfNxFaces) {
            const Index aux = dimensions.x() + 1;
            return Entity(grid, Coordinate(index % aux, (index / aux) % dimensions.y(), index / (aux * dimensions.y())),
                              Coordinate(1, 0, 0));
         }
         if (index < grid.numberOfNxAndNyFaces) {
            const Index i = index - grid.numberOfNxFaces;
            const Index aux = dimensions.y() + 1;
            return Entity(grid, Coordinate(i % dimensions.x(), (i / dimensions.x()) % aux, i / (aux * dimensions.x())),
                              Coordinate(0, 1, 0));
         }
         const Index i = index - grid.numberOfNxAndNyFaces;
         return Entity(grid, Coordinate(i % dimensions.x(), (i / dimensions.x()) % dimensions.y(), i / (dimensions.x() * dimensions.y())),
                           Coordinate(0, 0, 1));
      }

      __cuda_callable__ inline
      static Index getEntityIndex(const Grid& grid, const Entity& entity) {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         // TNL_ASSERT_LT(entity.getCoordinates(), grid.getDimensions() + abs(entity.getOrientation()), "wrong coordinates");

         const Coordinate coordinates = entity.getCoordinates();
         const Coordinate dimensions = grid.getDimensions();

         // if (entity.getOrientation().x()) {
         //    return (coordinates.z() * dimensions.y() + coordinates.y()) * (dimensions.x() + 1) + coordinates.x();
         // }
         // if (entity.getOrientation().y()) {
         //    return grid.numberOfNxFaces + (coordinates.z() * (dimensions.y() + 1) + coordinates.y()) * dimensions.x() + coordinates.x();
         // }
         return grid.numberOfNxAndNyFaces + (coordinates.z() * dimensions.y() + coordinates.y()) * dimensions.x() + coordinates.x();
      }
};

template <typename Real, typename Device, typename Index>
class GridEntityGetter<Meshes::Grid<3, Real, Device, Index>, 1> {
   public:
      static constexpr int EntityDimension = 1;

      using Grid = Meshes::Grid<3, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      static Entity getEntity(const Grid& grid, const Index& index) {
         TNL_ASSERT_GE(index, 0, "Index must be non-negative.");
         TNL_ASSERT_LT(index, grid.template getEntitiesCount<Entity>(), "Index is out of bounds.");

         const Coordinate dimensions = grid.getDimensions();

         if (index < grid.numberOfDxEdges) {
            const Index aux = dimensions.y() + 1;
            return Entity(grid, Coordinate(index % dimensions.x(), (index / dimensions.x()) % aux, index / (dimensions.x() * aux)),
                              Coordinate(0, 0, 0));
         }
         if (index < grid.numberOfDxAndDyEdges) {
            const Index i = index - grid.numberOfDxEdges;
            const Index aux = dimensions.x() + 1;
            return Entity(grid, Coordinate(i % aux, (i / aux) % dimensions.y(), i / (aux * dimensions.y())),
                              Coordinate(0, 0, 0));
         }
         const Index i = index - grid.numberOfDxAndDyEdges;
         const Index aux1 = dimensions.x() + 1;
         const Index aux2 = dimensions.y() + 1;
         return Entity(grid, Coordinate(i % aux1, (i / aux1) % aux2, i / (aux1 * aux2)), Coordinate(0, 0, 0));
      }

      __cuda_callable__ inline
      static Index getEntityIndex(const Grid& grid, const Entity& entity) {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         // TNL_ASSERT_LT(entity.getCoordinates(), grid.getDimensions() + Coordinate(1, 1, 1) - entity.getBasis(), "wrong coordinates");

         const Coordinate coordinates = entity.getCoordinates();
         const Coordinate dimensions = grid.getDimensions();

         // if (entity.getBasis().x()) return (coordinates.z() * (dimensions.y() + 1) + coordinates.y()) * dimensions.x() + coordinates.x();
         // if (entity.getBasis().y())
         //    return grid.numberOfDxEdges + (coordinates.z() * dimensions.y() + coordinates.y()) * (dimensions.x() + 1) + coordinates.x();
         return grid.numberOfDxAndDyEdges + (coordinates.z() * (dimensions.y() + 1) + coordinates.y()) * (dimensions.x() + 1) + coordinates.x();
      }
};

template <typename Real, typename Device, typename Index>
class GridEntityGetter<Meshes::Grid<3, Real, Device, Index>, 0> {
   public:
      static constexpr int EntityDimension = 0;

      using Grid = Meshes::Grid<3, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      static Entity getEntity(const Grid& grid, const Index& index) {
         TNL_ASSERT_GE(index, 0, "Index must be non-negative.");
         TNL_ASSERT_LT(index, grid.template getEntitiesCount<Entity>(), "Index is out of bounds.");

         const Coordinate dimensions = grid.getDimensions();

         const Index auxX = dimensions.x() + 1;
         const Index auxY = dimensions.y() + 1;
         return Entity(grid, Coordinate(index % auxX, (index / auxX) % auxY, index / (auxX * auxY)),
                           Coordinate(0, 0, 0));
      }

      __cuda_callable__ inline
      static Index getEntityIndex(const Grid& grid, const Entity& entity) {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LE(entity.getCoordinates(), grid.getDimensions(), "wrong coordinates");

         const Coordinate coordinates = entity.getCoordinates();
         const Coordinate dimensions = grid.getDimensions();

         return (coordinates.z() * (dimensions.y() + 1) + coordinates.y()) * (dimensions.x() + 1) + coordinates.x();
      }
};

}  // namespace Meshes
}  // namespace TNL
