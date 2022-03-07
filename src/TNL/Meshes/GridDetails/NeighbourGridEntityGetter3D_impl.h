/***************************************************************************
                          NeighbourGridEntityGetter3D_impl.h  -  description
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
 * | EntityDimenions | NeighbourEntityDimension  |
 * +-----------------+---------------------------+
 * |       3         |              3            |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<3, Real, Device, Index>, 3>, 3> {
   public:
      static constexpr int EntityDimension = 3;
      static constexpr int NeighbourEntityDimension = 3;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline NeighbourEntity getEntity() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(
            entity.getCoordinates() + Coordinate(stepX, stepY) >= Coordinate(0, 0, 0) &&
               entity.getCoordinates() + Coordinate(stepX, stepY) < entity.getMesh().getDimensions(),
            std::cerr << "entity.getCoordinates()  + Coordinate( stepX, stepY ) = " << entity.getCoordinates() + Coordinate(stepX, stepY)
                     << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return NeighbourEntity(this->entity.getMesh(), Coordinate(entity.getCoordinates().x() + stepX, entity.getCoordinates().y() + stepY,
                                                                              entity.getCoordinates().z() + stepZ));
      }

      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline Index getEntityIndex() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates() + Coordinate(stepX, stepY, stepZ) >= Coordinate(0, 0, 0) &&
                        entity.getCoordinates() + Coordinate(stepX, stepY, stepZ) < entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates()  + Coordinate( stepX, stepY, stepZ ) = "
                              << entity.getCoordinates() + Coordinate(stepX, stepY, stepZ)
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return this->entity.getIndex() + (stepZ * entity.getMesh().getDimensions().y() + stepY) * entity.getMesh().getDimensions().x() + stepX;
      }
   protected:
      const Entity& entity;
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimension  |
 * +-----------------+---------------------------+
 * |       3         |              2            |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<3, Real, Device, Index>, 3>, 2> {
   public:
      static constexpr int EntityDimension = 3;
      static constexpr int NeighbourEntityDimension = 2;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline
      NeighbourEntity getEntity() const {
         static_assert(!stepX + !stepY + !stepZ == 2, "Only one of the steps can be non-zero.");
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(
            entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0), stepZ + (stepZ < 0)) >= Coordinate(0, 0, 0) &&
               entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0), stepZ + (stepZ < 0)) <
                     entity.getMesh().getDimensions() + Coordinate((stepX > 0), (stepY > 0), (stepZ > 0)),
            std::cerr << "entity.getCoordinates()  + Coordinate( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) ) = "
                     << entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0), stepZ + (stepZ < 0))
                     << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return NeighbourEntity(
            this->entity.getMesh(),
            Coordinate(entity.getCoordinates().x() + stepX + (stepX < 0), entity.getCoordinates().y() + stepY + (stepY < 0),
                           entity.getCoordinates().z() + stepZ + (stepZ < 0)),
            Coordinate(stepX ? (stepX > 0 ? 1 : -1) : 0, stepY ? (stepY > 0 ? 1 : -1) : 0, stepZ ? (stepZ > 0 ? 1 : -1) : 0),
            Coordinate(!stepX, !stepY, !stepZ));
      }

      // TODO: - Fix
      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline
      Index getEntityIndex() const {
         return -1;//GridEntityGetterType::getEntityIndex(this->entity.getMesh(), getEntity<stepX, stepY, stepZ>());
      }

   protected:
      const Entity& entity;

   // NeighbourGridEntityGetter(){};
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimension  |
 * +-----------------+---------------------------+
 * |       3         |              1            |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<3, Real, Device, Index>, 3>, 1> {
   public:
      static constexpr int EntityDimension = 3;
      static constexpr int NeighbourEntityDimension = 1;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline
      NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline
      NeighbourEntity getEntity() const {
         static_assert(!stepX + !stepY + !stepZ == 1, "Exactly two of the steps must be non-zero.");
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(
            entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0), stepZ + (stepZ < 0)) >= Coordinate(0, 0, 0) &&
               entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0), stepZ + (stepZ < 0)) <
                     entity.getMesh().getDimensions() + Coordinate((stepX > 0), (stepY > 0), (stepZ > 0)),
            std::cerr << "entity.getCoordinates()  + Coordinate( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) ) = "
                     << entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0), stepZ + (stepZ < 0))
                     << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return NeighbourEntity(
            this->entity.getMesh(),
            Coordinate(entity.getCoordinates().x() + stepX + (stepX < 0), entity.getCoordinates().y() + stepY + (stepY < 0),
                           entity.getCoordinates().z() + stepZ + (stepZ < 0)),
            Coordinate(!!stepX, !!stepY, !!stepZ), Coordinate(!stepX, !stepY, !stepZ));
      }

      // TODO: - Fix
      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline
       Index getEntityIndex() const {
         return -1;//GridEntityGetterType::getEntityIndex(this->entity.getMesh(), getEntity<stepX, stepY, stepZ>());
      }

   protected:
      const Entity& entity;
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimension  |
 * +-----------------+---------------------------+
 * |       3         |            0              |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<3, Real, Device, Index>, 3>, 0> {
   public:
      static constexpr int EntityDimension = 3;
      static constexpr int NeighbourEntityDimension = 0;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline NeighbourEntity getEntity() const {
         TNL_ASSERT(stepX != 0 && stepY != 0 && stepZ != 0, std::cerr << " stepX = " << stepX << " stepY = " << stepY << " stepZ = " << stepZ);
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(
            entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0), stepZ + (stepZ < 0)) >= Coordinate(0, 0, 0) &&
               entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0), stepZ + (stepZ < 0)) <
                     entity.getMesh().getDimensions() + Coordinate((stepX > 0), (stepY > 0), (stepZ > 0)),
            std::cerr << "entity.getCoordinates()  + Coordinate( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 )  ) = "
                     << entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0), stepZ + (stepZ < 0))
                     << " entity.getMesh().getDimensions() + Coordinate( sign( stepX ), sign( stepY ), sign( stepZ ) ) = "
                     << entity.getMesh().getDimensions() + Coordinate(sign(stepX), sign(stepY), sign(stepZ))
                     << " EntityDimension = " << EntityDimension);
         return NeighbourEntity(this->entity.getMesh(), Coordinate(entity.getCoordinates().x() + stepX + (stepX < 0),
                                                                              entity.getCoordinates().y() + stepY + (stepY < 0),
                                                                              entity.getCoordinates().z() + stepZ + (stepZ < 0)));
      }

      // TODO: - Fix
      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline Index getEntityIndex() const {
         return -1;//GridEntityGetterType::getEntityIndex(entity.getMesh(), getEntity<stepX, stepY, stepZ>());
      }

   protected:
      const Entity& entity;
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimension  |
 * +-----------------+---------------------------+
 * |       2         |              3            |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<3, Real, Device, Index>, 2>, 3> {
   public:
      static constexpr int EntityDimension = 2;
      static constexpr int NeighbourEntityDimension = 3;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline NeighbourEntity getEntity() const {
         /*TNL_ASSERT( ( ( !! stepX ) == ( !! entity.getOrientation().x() ) ) &&
                  ( ( !! stepY ) == ( !! entity.getOrientation().y() ) ) &&
                  ( ( !! stepZ ) == ( !! entity.getOrientation().z() ) ),
                  std::cerr << "( stepX, stepY, stepZ ) cannot be perpendicular to entity coordinates: stepX = " << stepX
                        << " stepY = " << stepY << " stepZ = " << stepZ
                        << " entity.getOrientation() = " << entity.getOrientation() );*/
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions() + TNL::abs(entity.getOrientation()), "wrong coordinates");
         TNL_ASSERT(
            entity.getCoordinates() + Coordinate(stepX - (stepX > 0) * (entity.getOrientation().x() != 0.0),
                                                      stepY - (stepY > 0) * (entity.getOrientation().y() != 0.0),
                                                      stepZ - (stepZ > 0) * (entity.getOrientation().z() != 0.0)) >=
                     Coordinate(0, 0, 0) &&
               entity.getCoordinates() + Coordinate(stepX - (stepX > 0) * (entity.getOrientation().x() != 0.0),
                                                         stepY - (stepY > 0) * (entity.getOrientation().y() != 0.0),
                                                         stepZ - (stepZ > 0) * (entity.getOrientation().z() != 0.0)) <
                     entity.getMesh().getDimensions(),
            std::cerr << "entity.getCoordinates()  + Coordinate( stepX + ( stepX < 0 ) * ( entity.getOrientation().x() != 0.0 ), stepY + ( stepY "
                        "< 0 ) * ( entity.getOrientation().y() != 0.0 ), stepZ + ( stepZ < 0 ) * ( entity.getOrientation().z() != 0.0 ) ) = "
                     << entity.getCoordinates() + Coordinate(stepX + (stepX < 0) * (entity.getOrientation().x() != 0.0),
                                                                  stepY + (stepY < 0) * (entity.getOrientation().y() != 0.0),
                                                                  stepZ + (stepZ < 0) * (entity.getOrientation().z() != 0.0))
                     << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return NeighbourEntity(this->entity.getMesh(),
                                       Coordinate(entity.getCoordinates().x() + stepX - (stepX > 0) * (entity.getOrientation().x() != 0.0),
                                                      entity.getCoordinates().y() + stepY - (stepY > 0) * (entity.getOrientation().y() != 0.0),
                                                      entity.getCoordinates().z() + stepZ - (stepZ > 0) * (entity.getOrientation().z() != 0.0)));
      }

      // TODO: - Fix
      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline Index getEntityIndex() const {
         return -1;//GridEntityGetterType::getEntityIndex(entity.getMesh(), getEntity<stepX, stepY, stepZ>());
      }

   protected:
      const Entity& entity;
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighbourEntityDimension  |
 * +-----------------+---------------------------+
 * |       0         |              0            |
 * +-----------------+---------------------------+
 */
template <typename Real, typename Device, typename Index>
class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<3, Real, Device, Index>, 0>, 0> {
   public:
      static constexpr int EntityDimension = 0;
      static constexpr int NeighbourEntityDimension = 0;

      using Grid = Meshes::Grid<1, Real, Device, Index>;
      using Entity = GridEntity<Grid, EntityDimension>;
      using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
      using Coordinate = typename Grid::Coordinate;

      __cuda_callable__ inline NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline NeighbourEntity getEntity() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates() + Coordinate(stepX, stepY, stepZ) >= Coordinate(0, 0, 0) &&
                        entity.getCoordinates() + Coordinate(stepX, stepY, stepZ) <= entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates()  + Coordinate( stepX, stepY, stepZ ) = "
                              << entity.getCoordinates() + Coordinate(stepX, stepY, stepZ)
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return NeighbourEntity(this->entity.getMesh(), Coordinate(entity.getCoordinates().x() + stepX, entity.getCoordinates().y() + stepY,
                                                                              entity.getCoordinates().z() + stepZ));
      }

      template <int stepX, int stepY, int stepZ>
      __cuda_callable__ inline Index getEntityIndex() const {
         TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0, 0), "wrong coordinates");
         TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
         TNL_ASSERT(entity.getCoordinates() + Coordinate(stepX, stepY, stepZ) >= Coordinate(0, 0, 0) &&
                        entity.getCoordinates() + Coordinate(stepX, stepY, stepZ) <= entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates()  + Coordinate( stepX, stepY, stepZ ) = "
                              << entity.getCoordinates() + Coordinate(stepX, stepY, stepZ)
                              << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
         return this->entity.getIndex() + stepZ * (entity.getMesh().getDimensions().y() + 1 + stepY) * (entity.getMesh().getDimensions().x() + 1) +
               stepX;
      }

   protected:
      const Entity& entity;
};

}  // namespace Meshes
}  // namespace TNL
