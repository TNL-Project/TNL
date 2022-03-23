/***************************************************************************
                          NeighbourGridEntityGetter2D_impl.h  -  description
                             -------------------
    begin                : Nov 23, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

// #include <TNL/Meshes/GridDetails/Grid1D.h>
// #include <TNL/Meshes/GridDetails/Grid2D.h>
// #include <TNL/Meshes/GridDetails/Grid3D.h>
// #include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter.h>

// namespace TNL {
// namespace Meshes {

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighbourEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       2         |              2            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 2>, 2> {
//    public:
//       static constexpr int EntityDimension = 2;
//       static constexpr int NeighbourEntityDimension = 2;

//       using Grid = Meshes::Grid<1, Real, Device, Index>;
//       using Entity = GridEntity<Grid, EntityDimension>;
//       using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
//       using Coordinate = typename Grid::Coordinate;

//       __cuda_callable__ inline
//       NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//        NeighbourEntity getEntity() const {
//          TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
//          TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//          TNL_ASSERT(
//             entity.getCoordinates() + Coordinate(stepX, stepY) >= Coordinate(0, 0) &&
//                entity.getCoordinates() + Coordinate(stepX, stepY) < entity.getMesh().getDimensions(),
//             std::cerr << "entity.getCoordinates()  + Coordinate( stepX, stepY ) = " << entity.getCoordinates() + Coordinate(stepX, stepY)
//                      << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//          return NeighbourEntity(this->entity.getMesh(),
//                                 Coordinate(entity.getCoordinates().x() + stepX, entity.getCoordinates().y() + stepY));
//       }

//       // TODO: - Fix
//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//       Index getEntityIndex() const {
//          TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
//          TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//          TNL_ASSERT(
//             entity.getCoordinates() + Coordinate(stepX, stepY) >= Coordinate(0, 0) &&
//                entity.getCoordinates() + Coordinate(stepX, stepY) < entity.getMesh().getDimensions(),
//             std::cerr << "entity.getCoordinates()  + Coordinate( stepX, stepY ) = " << entity.getCoordinates() + Coordinate(stepX, stepY)
//                      << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//          return this->entity.getIndex() + stepY * entity.getMesh().getDimensions().x() + stepX;
//       }
//    protected:
//       const Entity& entity;
// };

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighbourEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       2         |              1            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 2>, 1> {
//    public:
//       static constexpr int EntityDimension = 2;
//       static constexpr int NeighbourEntityDimension = 1;

//       using Grid = Meshes::Grid<1, Real, Device, Index>;
//       using Entity = GridEntity<Grid, EntityDimension>;
//       using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
//       using Coordinate = typename Grid::Coordinate;

//       __cuda_callable__ inline
//       NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//       NeighbourEntity getEntity() const {
//          static_assert(!stepX + !stepY == 1, "Only one of the steps can be non-zero.");
//          TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
//          TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//          TNL_ASSERT(entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0)) >= Coordinate(0, 0) &&
//                         entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0)) <
//                            entity.getMesh().getDimensions() + Coordinate((stepX > 0), (stepY > 0)),
//                   std::cerr << "entity.getCoordinates()  + Coordinate( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
//                               << entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0))
//                               << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);

//          return NeighbourEntity(
//             this->entity.getMesh(),
//             Coordinate(entity.getCoordinates().x() + stepX + (stepX < 0), entity.getCoordinates().y() + stepY + (stepY < 0)),
//             Coordinate(stepX ? (stepX > 0 ? 1 : -1) : 0, stepY ? (stepY > 0 ? 1 : -1) : 0), Coordinate(!stepX, !stepY));
//       }

//       // TODO: - Fix
//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//       Index getEntityIndex() const {
//          return -1;// GridEntityGetterType::getEntityIndex(this->entity.getMesh(), this->template getEntity<stepX, stepY>());
//       }
//    protected:
//       const Entity& entity;
// };

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighbourEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       2         |            0              |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 2>, 0> {
//    public:
//       static constexpr int EntityDimension = 2;
//       static constexpr int NeighbourEntityDimension = 0;

//       using Grid = Meshes::Grid<1, Real, Device, Index>;
//       using Entity = GridEntity<Grid, EntityDimension>;
//       using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
//       using Coordinate = typename Grid::Coordinate;

//       __cuda_callable__ inline
//       NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//       NeighbourEntity getEntity() const {
//          TNL_ASSERT(stepX != 0 && stepY != 0, std::cerr << " stepX = " << stepX << " stepY = " << stepY);
//          TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
//          TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//          TNL_ASSERT(entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0)) >= Coordinate(0, 0) &&
//                         entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0)) <
//                            entity.getMesh().getDimensions() + Coordinate((stepX > 0), (stepY > 0)),
//                   std::cerr << "entity.getCoordinates()  + Coordinate( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
//                               << entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0))
//                               << " entity.getMesh().getDimensions() + Coordinate( sign( stepX ), sign( stepY ) ) = "
//                               << entity.getMesh().getDimensions() + Coordinate(sign(stepX), sign(stepY))
//                               << " EntityDimension = " << EntityDimension);
//          return NeighbourEntity(this->entity.getMesh(), Coordinate(entity.getCoordinates().x() + stepX + (stepX < 0),
//                                                                               entity.getCoordinates().y() + stepY + (stepY < 0)));
//       }

//       // TODO: - Fix
//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//        Index getEntityIndex() const {
//          return -1;//GridEntityGetterType::getEntityIndex(this->entity.getMesh(), this->template getEntity<stepX, stepY>());
//       }
//    protected:
//       const Entity& entity;
// };

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighbourEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       1         |              2            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 1>, 2> {
//    public:
//       static constexpr int EntityDimension = 1;
//       static constexpr int NeighbourEntityDimension = 2;

//       using Grid = Meshes::Grid<1, Real, Device, Index>;
//       using Entity = GridEntity<Grid, EntityDimension>;
//       using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
//       using Coordinate = typename Grid::Coordinate;

//       __cuda_callable__ inline
//       NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//       NeighbourEntity getEntity() const {
//          /*TNL_ASSERT( ( ( !! stepX ) == ( !! entity.getOrientation().x() ) ) &&
//                   ( ( !! stepY ) == ( !! entity.getOrientation().y() ) ),
//                   std::cerr << "( stepX, stepY ) cannot be perpendicular to entity coordinates: stepX = " << stepX << " stepY = " << stepY
//                         << " entity.getOrientation() = " << entity.getOrientation() );*/
//          TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
//          TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions() + TNL::abs(entity.getOrientation()), "wrong coordinates");
//          TNL_ASSERT(entity.getCoordinates() + Coordinate(stepX - (stepX > 0) * (entity.getOrientation().x() != 0.0),
//                                                             stepY - (stepY > 0) * (entity.getOrientation().y() != 0.0)) >=
//                            Coordinate(0, 0) &&
//                         entity.getCoordinates() + Coordinate(stepX - (stepX > 0) * (entity.getOrientation().x() != 0.0),
//                                                                   stepY - (stepY > 0) * (entity.getOrientation().y() != 0.0)) <
//                            entity.getMesh().getDimensions(),
//                   std::cerr << "entity.getCoordinates()  + Coordinate( stepX + ( stepX < 0 )  * ( entity.getOrientation().x() != 0.0 ), stepY + "
//                                  "( stepY < 0 ) * ( entity.getOrientation().y() != 0.0 ) ) = "
//                               << entity.getCoordinates() + Coordinate(stepX + (stepX < 0), stepY + (stepY < 0))
//                               << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//          return NeighbourEntity(this->entity.getMesh(),
//                                        Coordinate(entity.getCoordinates().x() + stepX - (stepX > 0) * (entity.getOrientation().x() != 0.0),
//                                                       entity.getCoordinates().y() + stepY - (stepY > 0) * (entity.getOrientation().y() != 0.0)));
//       }

//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//        Index getEntityIndex() const {
//          return -1;// GridEntityGetterType::getEntityIndex(this->entity.getMesh(), this->template getEntity<stepX, stepY>());
//       }
//    protected:
//       const Entity& entity;
// };

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighbourEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       0         |              0            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighbourGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 0>, 0> {
//    public:
//       static constexpr int EntityDimension = 0;
//       static constexpr int NeighbourEntityDimension = 0;

//       using Grid = Meshes::Grid<1, Real, Device, Index>;
//       using Entity = GridEntity<Grid, EntityDimension>;
//       using NeighbourEntity = GridEntity<Grid, NeighbourEntityDimension>;
//       using Coordinate = typename Grid::Coordinate;

//       __cuda_callable__ inline
//       NeighbourGridEntityGetter(const Entity& entity) : entity(entity) {}

//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//       NeighbourEntity getEntity() const {
//          TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
//          TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//          TNL_ASSERT(
//             entity.getCoordinates() + Coordinate(stepX, stepY) >= Coordinate(0, 0) &&
//                entity.getCoordinates() + Coordinate(stepX, stepY) <= entity.getMesh().getDimensions(),
//             std::cerr << "entity.getCoordinates()  + Coordinate( stepX, stepY ) = " << entity.getCoordinates() + Coordinate(stepX, stepY)
//                      << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//          return NeighbourEntity(this->entity.getMesh(),
//                                        Coordinate(entity.getCoordinates().x() + stepX, entity.getCoordinates().y() + stepY));
//       }

//       template <int stepX, int stepY>
//       __cuda_callable__ inline
//       Index getEntityIndex() const {
//          TNL_ASSERT_GE(entity.getCoordinates(), Coordinate(0, 0), "wrong coordinates");
//          TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//          TNL_ASSERT(
//             entity.getCoordinates() + Coordinate(stepX, stepY) >= Coordinate(0, 0) &&
//                entity.getCoordinates() + Coordinate(stepX, stepY) <= entity.getMesh().getDimensions(),
//             std::cerr << "entity.getCoordinates()  + Coordinate( stepX, stepY ) = " << entity.getCoordinates() + Coordinate(stepX, stepY)
//                      << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);

//          return this->entity.getIndex() + stepY * (entity.getMesh().getDimensions().x() + 1) + stepX;
//       }
//    protected:
//       const Entity& entity;
// };

// }  // namespace Meshes
// }  // namespace TNL
