/***************************************************************************
                          NeighbourGridEntityGetter1D_impl.h  -  description
                             -------------------
    begin                : Nov 23, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter.h>

namespace TNL {
namespace Meshes {

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimension   |
 * +-----------------+---------------------------+
 * |       1         |              1            |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<1, Real, Device, Index>, 1>, 1> {
   public:
      static constexpr int EntityDimension = 1;
      static constexpr int NeighbourEntityDimension = 1;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int step>
      __cuda_callable__ inline
      NeighbourEntity getEntity() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates() + Coordinate(step) >= Coordinate(0) &&
                    entity.getCoordinates() + Coordinate(step) < entity.getMesh().getDimensions(),
                 std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                           << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);

      return NeighbourEntity(Coordinate(entity.getCoordinates().x() + step));
   }

      template <int step>
      __cuda_callable__ inline
      Index getEntityIndex() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates() + Coordinate(step) >= Coordinate(0) &&
                        entity.getCoordinates() + Coordinate(step) < entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);

         return this->entity.getIndex() + step;
      }
   protected:
      const Entity& entity;
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimension   |
 * +-----------------+---------------------------+
 * |       1         |              0            |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<1, Real, Device, Index>, 1>, 0> {
   public:
      static constexpr int EntityDimension = 1;
      static constexpr int NeighbourEntityDimension = 0;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int step>
      __cuda_callable__ inline
      NeighbourEntity getEntity() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates().x() + step + (step < 0) >= Coordinate(0) &&
                        entity.getCoordinates().x() + step + (step < 0) <= entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return NeighbourEntity(Coordinate(entity.getCoordinates().x() + step + (step < 0)));
      }

      template <int step>
      __cuda_callable__ inline
      Index getEntityIndex() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates().x() + step + (step < 0) >= Coordinate(0).x() &&
                        entity.getCoordinates().x() + step + (step < 0) <= entity.getMesh().getDimensions().x(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return this->entity.getIndex() + step + (step < 0);
      }
   protected:
      const Entity& entity;
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimension   |
 * +-----------------+---------------------------+
 * |       0         |              1            |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<1, Real, Device, Index>, 0>, 1> {
   public:
      static constexpr int EntityDimension = 0;
      static constexpr int NeighbourEntityDimension = 1;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int step>
      __cuda_callable__ inline
      NeighbourEntity getEntity() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0), "wrong coordinates");
         TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates().x() + step - (step > 0) >= Coordinate(0) &&
                        entity.getCoordinates().x() + step - (step > 0) < entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return NeighbourEntity(Coordinate(entity.getCoordinates().x() + step - (step > 0)));
      }

      template <int step>
      __cuda_callable__ inline
      Index getEntityIndex() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0), "wrong coordinates");
         TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates().x() + step - (step > 0) >= 0 &&
                        entity.getCoordinates().x() + step - (step > 0) < entity.getMesh().getDimensions().x(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return this->entity.getIndex() + step - (step > 0);
      }
   protected:
      const Entity& entity;
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimension   |
 * +-----------------+---------------------------+
 * |       0         |              0            |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<1, Real, Device, Index>, 0>, 0> {
   public:
      static constexpr int EntityDimension = 0;
      static constexpr int NeighbourEntityDimension = 0;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int step>
      __cuda_callable__ inline
      NeighbourEntity getEntity() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0), "wrong coordinates");
         TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates().x() + step >= Coordinate(0) && entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return NeighbourEntity(Coordinate(entity.getCoordinates().x() + step));
      }

      template <int step>
      __cuda_callable__ inline
      Index getEntityIndex() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0), "wrong coordinates");
         TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates().x() + step >= Coordinate(0) && entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);

         return this->entity.getIndex() + step;
      }

   protected:
      const Entity& entity;
};

}  // namespace Meshes
}  // namespace TNL
