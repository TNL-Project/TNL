// /***************************************************************************
//                           NeighborGridEntityGetter2D_impl.h  -  description
//                              -------------------
//     begin                : Nov 23, 2015
//     copyright            : (C) 2015 by Tomas Oberhuber
//     email                : tomas.oberhuber@fjfi.cvut.cz
//  ***************************************************************************/

// /* See Copyright Notice in tnl/Copyright */

// #pragma once

// #include <TNL/Meshes/GridDetails/Grid1D.h>
// #include <TNL/Meshes/GridDetails/Grid2D.h>
// #include <TNL/Meshes/GridDetails/Grid3D.h>
// #include <TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>

// namespace TNL {
// namespace Meshes {

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighborEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       2         |              2            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighborGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 2>, 2> {
//   public:
//    static constexpr int EntityDimension = 2;
//    static constexpr int NeighborEntityDimension = 2;
//    typedef Meshes::Grid<2, Real, Device, Index> GridType;
//    typedef GridEntity<GridType, EntityDimension, Config> GridEntityType;
//    typedef GridEntity<GridType, NeighborEntityDimension, Config> NeighborGridEntityType;
//    typedef Real RealType;
//    typedef Index IndexType;
//    typedef typename GridType::CoordinatesType CoordinatesType;
//    typedef GridEntityGetter<GridType, NeighborGridEntityType> GridEntityGetterType;

//    __cuda_callable__ inline NeighborGridEntityGetter(const GridEntityType& entity) : entity(entity) {}

//    template <int stepX, int stepY>
//    __cuda_callable__ inline NeighborGridEntityType getEntity() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0, 0), "wrong coordinates");
//       TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(
//           entity.getCoordinates() + CoordinatesType(stepX, stepY) >= CoordinatesType(0, 0) &&
//               entity.getCoordinates() + CoordinatesType(stepX, stepY) < entity.getMesh().getDimensions(),
//           std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates() + CoordinatesType(stepX, stepY)
//                     << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return NeighborGridEntityType(this->entity.getMesh(),
//                                     CoordinatesType(entity.getCoordinates().x() + stepX, entity.getCoordinates().y() + stepY));
//    }

//    template <int stepX, int stepY>
//    __cuda_callable__ inline IndexType getEntityIndex() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0, 0), "wrong coordinates");
//       TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(
//           entity.getCoordinates() + CoordinatesType(stepX, stepY) >= CoordinatesType(0, 0) &&
//               entity.getCoordinates() + CoordinatesType(stepX, stepY) < entity.getMesh().getDimensions(),
//           std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates() + CoordinatesType(stepX, stepY)
//                     << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return this->entity.getIndex() + stepY * entity.getMesh().getDimensions().x() + stepX;
//    }

//    __cuda_callable__ void refresh(const GridType& grid, const IndexType& entityIndex){};

//   protected:
//    const GridEntityType& entity;

//    // NeighborGridEntityGetter(){};
// };

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighborEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       2         |              1            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighborGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 2>, 1> {
//   public:
//    static constexpr int EntityDimension = 2;
//    static constexpr int NeighborEntityDimension = 1;
//    typedef Meshes::Grid<2, Real, Device, Index> GridType;
//    typedef GridEntity<GridType, EntityDimension, Config> GridEntityType;
//    typedef GridEntity<GridType, NeighborEntityDimension, Config> NeighborGridEntityType;
//    typedef Real RealType;
//    typedef Index IndexType;
//    typedef typename GridType::CoordinatesType CoordinatesType;
//    typedef GridEntityGetter<GridType, NeighborGridEntityType> GridEntityGetterType;
//    typedef typename GridEntityType::EntityOrientationType EntityOrientationType;
//    typedef typename GridEntityType::EntityBasisType EntityBasisType;

//    __cuda_callable__ inline NeighborGridEntityGetter(const GridEntityType& entity) : entity(entity) {}

//    template <int stepX, int stepY>
//    __cuda_callable__ inline NeighborGridEntityType getEntity() const {
//       static_assert(!stepX + !stepY == 1, "Only one of the steps can be non-zero.");
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0, 0), "wrong coordinates");
//       TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates() + CoordinatesType(stepX + (stepX < 0), stepY + (stepY < 0)) >= CoordinatesType(0, 0) &&
//                      entity.getCoordinates() + CoordinatesType(stepX + (stepX < 0), stepY + (stepY < 0)) <
//                          entity.getMesh().getDimensions() + CoordinatesType((stepX > 0), (stepY > 0)),
//                  std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
//                            << entity.getCoordinates() + CoordinatesType(stepX + (stepX < 0), stepY + (stepY < 0))
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return NeighborGridEntityType(
//           this->entity.getMesh(),
//           CoordinatesType(entity.getCoordinates().x() + stepX + (stepX < 0), entity.getCoordinates().y() + stepY + (stepY < 0)),
//           EntityOrientationType(stepX ? (stepX > 0 ? 1 : -1) : 0, stepY ? (stepY > 0 ? 1 : -1) : 0), EntityBasisType(!stepX, !stepY));
//    }

//    template <int stepX, int stepY>
//    __cuda_callable__ inline IndexType getEntityIndex() const {
//       return GridEntityGetterType::getEntityIndex(this->entity.getMesh(), this->template getEntity<stepX, stepY>());
//    }

//    __cuda_callable__ void refresh(const GridType& grid, const IndexType& entityIndex){};

//   protected:
//    const GridEntityType& entity;
// };

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighborEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       2         |            0              |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighborGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 2>, 0> {
//   public:
//    static constexpr int EntityDimension = 2;
//    static constexpr int NeighborEntityDimension = 0;
//    typedef Meshes::Grid<2, Real, Device, Index> GridType;
//    typedef GridEntity<GridType, EntityDimension, Config> GridEntityType;
//    typedef GridEntity<GridType, NeighborEntityDimension, Config> NeighborGridEntityType;
//    typedef Real RealType;
//    typedef Index IndexType;
//    typedef typename GridType::CoordinatesType CoordinatesType;
//    typedef GridEntityGetter<GridType, NeighborGridEntityType> GridEntityGetterType;

//    __cuda_callable__ inline NeighborGridEntityGetter(const GridEntityType& entity) : entity(entity) {}

//    template <int stepX, int stepY>
//    __cuda_callable__ inline NeighborGridEntityType getEntity() const {
//       TNL_ASSERT(stepX != 0 && stepY != 0, std::cerr << " stepX = " << stepX << " stepY = " << stepY);
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0, 0), "wrong coordinates");
//       TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates() + CoordinatesType(stepX + (stepX < 0), stepY + (stepY < 0)) >= CoordinatesType(0, 0) &&
//                      entity.getCoordinates() + CoordinatesType(stepX + (stepX < 0), stepY + (stepY < 0)) <
//                          entity.getMesh().getDimensions() + CoordinatesType((stepX > 0), (stepY > 0)),
//                  std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
//                            << entity.getCoordinates() + CoordinatesType(stepX + (stepX < 0), stepY + (stepY < 0))
//                            << " entity.getMesh().getDimensions() + CoordinatesType( sign( stepX ), sign( stepY ) ) = "
//                            << entity.getMesh().getDimensions() + CoordinatesType(sign(stepX), sign(stepY))
//                            << " EntityDimension = " << EntityDimension);
//       return NeighborGridEntityType(this->entity.getMesh(), CoordinatesType(entity.getCoordinates().x() + stepX + (stepX < 0),
//                                                                             entity.getCoordinates().y() + stepY + (stepY < 0)));
//    }

//    template <int stepX, int stepY>
//    __cuda_callable__ inline IndexType getEntityIndex() const {
//       return GridEntityGetterType::getEntityIndex(this->entity.getMesh(), this->template getEntity<stepX, stepY>());
//    }

//    __cuda_callable__ void refresh(const GridType& grid, const IndexType& entityIndex){};

//   protected:
//    const GridEntityType& entity;

//    // NeighborGridEntityGetter(){};
// };

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighborEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       1         |              2            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighborGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 1>, 2> {
//   public:
//    static constexpr int EntityDimension = 1;
//    static constexpr int NeighborEntityDimension = 2;
//    typedef Meshes::Grid<2, Real, Device, Index> GridType;
//    typedef GridEntity<GridType, EntityDimension, Config> GridEntityType;
//    typedef GridEntity<GridType, NeighborEntityDimension, Config> NeighborGridEntityType;
//    typedef Real RealType;
//    typedef Index IndexType;
//    typedef typename GridType::CoordinatesType CoordinatesType;
//    typedef GridEntityGetter<GridType, NeighborGridEntityType> GridEntityGetterType;

//    __cuda_callable__ inline NeighborGridEntityGetter(const GridEntityType& entity) : entity(entity) {}

//    template <int stepX, int stepY>
//    __cuda_callable__ inline NeighborGridEntityType getEntity() const {
//       /*TNL_ASSERT( ( ( !! stepX ) == ( !! entity.getOrientation().x() ) ) &&
//                  ( ( !! stepY ) == ( !! entity.getOrientation().y() ) ),
//                  std::cerr << "( stepX, stepY ) cannot be perpendicular to entity coordinates: stepX = " << stepX << " stepY = " << stepY
//                       << " entity.getOrientation() = " << entity.getOrientation() );*/
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0, 0), "wrong coordinates");
//       TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions() + TNL::abs(entity.getOrientation()), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates() + CoordinatesType(stepX - (stepX > 0) * (entity.getOrientation().x() != 0.0),
//                                                            stepY - (stepY > 0) * (entity.getOrientation().y() != 0.0)) >=
//                          CoordinatesType(0, 0) &&
//                      entity.getCoordinates() + CoordinatesType(stepX - (stepX > 0) * (entity.getOrientation().x() != 0.0),
//                                                                stepY - (stepY > 0) * (entity.getOrientation().y() != 0.0)) <
//                          entity.getMesh().getDimensions(),
//                  std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 )  * ( entity.getOrientation().x() != 0.0 ), stepY + "
//                               "( stepY < 0 ) * ( entity.getOrientation().y() != 0.0 ) ) = "
//                            << entity.getCoordinates() + CoordinatesType(stepX + (stepX < 0), stepY + (stepY < 0))
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return NeighborGridEntityType(this->entity.getMesh(),
//                                     CoordinatesType(entity.getCoordinates().x() + stepX - (stepX > 0) * (entity.getOrientation().x() != 0.0),
//                                                     entity.getCoordinates().y() + stepY - (stepY > 0) * (entity.getOrientation().y() != 0.0)));
//    }

//    template <int stepX, int stepY>
//    __cuda_callable__ inline IndexType getEntityIndex() const {
//       return GridEntityGetterType::getEntityIndex(this->entity.getMesh(), this->template getEntity<stepX, stepY>());
//    }

//    __cuda_callable__ void refresh(const GridType& grid, const IndexType& entityIndex){};

//   protected:
//    const GridEntityType& entity;
// };

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighborEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       0         |              0            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighborGridEntityGetter<GridEntity<Meshes::Grid<2, Real, Device, Index>, 0>, 0> {
//   public:
//    static constexpr int EntityDimension = 0;
//    static constexpr int NeighborEntityDimension = 0;
//    typedef Meshes::Grid<2, Real, Device, Index> GridType;
//    typedef GridEntity<GridType, EntityDimension> GridEntityType;
//    typedef GridEntity<GridType, NeighborEntityDimension> NeighborGridEntityType;
//    typedef Real RealType;
//    typedef Index IndexType;
//    typedef typename GridType::CoordinatesType CoordinatesType;
//    typedef GridEntityGetter<GridType, NeighborGridEntityType> GridEntityGetterType;

//    __cuda_callable__ inline NeighborGridEntityGetter(const GridEntityType& entity) : entity(entity) {}

//    template <int stepX, int stepY>
//    __cuda_callable__ inline NeighborGridEntityType getEntity() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0, 0), "wrong coordinates");
//       TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(
//           entity.getCoordinates() + CoordinatesType(stepX, stepY) >= CoordinatesType(0, 0) &&
//               entity.getCoordinates() + CoordinatesType(stepX, stepY) <= entity.getMesh().getDimensions(),
//           std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates() + CoordinatesType(stepX, stepY)
//                     << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return NeighborGridEntityType(this->entity.getMesh(),
//                                     CoordinatesType(entity.getCoordinates().x() + stepX, entity.getCoordinates().y() + stepY));
//    }

//    template <int stepX, int stepY>
//    __cuda_callable__ inline IndexType getEntityIndex() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0, 0), "wrong coordinates");
//       TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(
//           entity.getCoordinates() + CoordinatesType(stepX, stepY) >= CoordinatesType(0, 0) &&
//               entity.getCoordinates() + CoordinatesType(stepX, stepY) <= entity.getMesh().getDimensions(),
//           std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates() + CoordinatesType(stepX, stepY)
//                     << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return this->entity.getIndex() + stepY * (entity.getMesh().getDimensions().x() + 1) + stepX;
//    }

//    __cuda_callable__ void refresh(const GridType& grid, const IndexType& entityIndex){};

//   protected:
//    const GridEntityType& entity;

//    // NeighborGridEntityGetter(){};
// };

// }  // namespace Meshes
// }  // namespace TNL
