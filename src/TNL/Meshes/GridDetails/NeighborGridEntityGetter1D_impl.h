// /***************************************************************************
//                           NeighborGridEntityGetter1D_impl.h  -  description
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
//  * |       1         |              1            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index>
// class NeighborGridEntityGetter<GridEntity<Meshes::Grid<1, Real, Device, Index>, 1>, 1> {
//    public:
//       static constexpr int EntityDimension = 1;
//       static constexpr int NeighborEntityDimension = 1;
//       typedef Meshes::Grid<1, Real, Device, Index> GridType;
//       typedef GridEntity<GridType, EntityDimension, Config> GridEntityType;
//       typedef GridEntity<GridType, NeighborEntityDimension, Config> NeighborGridEntityType;
//       typedef Real RealType;
//       typedef Index IndexType;
//       typedef typename GridType::CoordinatesType CoordinatesType;
//       typedef GridEntityGetter<GridType, NeighborGridEntityType> GridEntityGetterType;

//       __cuda_callable__ inline NeighborGridEntityGetter(const GridEntityType& entity) : entity(entity) {}

//       template <int step>
//       __cuda_callable__ inline NeighborGridEntityType getEntity() const {
//          TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0), "wrong coordinates");
//          TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//          TNL_ASSERT(entity.getCoordinates() + CoordinatesType(step) >= CoordinatesType(0) &&
//                     entity.getCoordinates() + CoordinatesType(step) < entity.getMesh().getDimensions(),
//                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);

//       return NeighborGridEntity(CoordinatesType(entity.getCoordinates().x() + step));
//    }

//    template <int step>
//    __cuda_callable__ inline IndexType getEntityIndex() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0), "wrong coordinates");
//       TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates() + CoordinatesType(step) >= CoordinatesType(0) &&
//                      entity.getCoordinates() + CoordinatesType(step) < entity.getMesh().getDimensions(),
//                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return this->entity.getIndex() + step;
//    }

//    __cuda_callable__ void refresh(const GridType& grid, const IndexType& entityIndex){};

//   protected:
//    const GridEntityType& entity;
// };

// /****
//  * +-----------------+---------------------------+
//  * | EntityDimenions | NeighborEntityDimension   |
//  * +-----------------+---------------------------+
//  * |       1         |              0            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index, typename Config>
// class NeighborGridEntityGetter<GridEntity<Meshes::Grid<1, Real, Device, Index>, 1>, 0> {
//   public:
//    static constexpr int EntityDimension = 1;
//    static constexpr int NeighborEntityDimension = 0;
//    typedef Meshes::Grid<1, Real, Device, Index> GridType;
//    typedef GridEntity<GridType, EntityDimension, Config> GridEntityType;
//    typedef GridEntity<GridType, NeighborEntityDimension, Config> NeighborGridEntityType;
//    typedef Real RealType;
//    typedef Index IndexType;
//    typedef typename GridType::CoordinatesType CoordinatesType;
//    typedef GridEntityGetter<GridType, NeighborGridEntityType> GridEntityGetterType;

//    __cuda_callable__ inline NeighborGridEntityGetter(const GridEntityType& entity) : entity(entity) {}

//    template <int step>
//    __cuda_callable__ inline NeighborGridEntityType getEntity() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0), "wrong coordinates");
//       TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates().x() + step + (step < 0) >= CoordinatesType(0) &&
//                      entity.getCoordinates().x() + step + (step < 0) <= entity.getMesh().getDimensions(),
//                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return NeighborGridEntity(CoordinatesType(entity.getCoordinates().x() + step + (step < 0)));
//    }

//    template <int step>
//    __cuda_callable__ inline IndexType getEntityIndex() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0), "wrong coordinates");
//       TNL_ASSERT_LT(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates().x() + step + (step < 0) >= CoordinatesType(0).x() &&
//                      entity.getCoordinates().x() + step + (step < 0) <= entity.getMesh().getDimensions().x(),
//                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return this->entity.getIndex() + step + (step < 0);
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
//  * |       0         |              1            |
//  * +-----------------+---------------------------+
//  */
// template <typename Real, typename Device, typename Index, typename Config>
// class NeighborGridEntityGetter<GridEntity<Meshes::Grid<1, Real, Device, Index>, 0>, 1> {
//   public:
//    static constexpr int EntityDimension = 0;
//    static constexpr int NeighborEntityDimension = 1;
//    typedef Meshes::Grid<1, Real, Device, Index> GridType;
//    typedef GridEntity<GridType, EntityDimension, Config> GridEntityType;
//    typedef GridEntity<GridType, NeighborEntityDimension, Config> NeighborGridEntityType;
//    typedef Real RealType;
//    typedef Index IndexType;
//    typedef typename GridType::CoordinatesType CoordinatesType;
//    typedef GridEntityGetter<GridType, NeighborGridEntityType> GridEntityGetterType;

//    __cuda_callable__ inline NeighborGridEntityGetter(const GridEntityType& entity) : entity(entity) {}

//    void test() const { std::cerr << "***" << std::endl; };

//    template <int step>
//    __cuda_callable__ inline NeighborGridEntityType getEntity() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0), "wrong coordinates");
//       TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates().x() + step - (step > 0) >= CoordinatesType(0) &&
//                      entity.getCoordinates().x() + step - (step > 0) < entity.getMesh().getDimensions(),
//                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return NeighborGridEntity(CoordinatesType(entity.getCoordinates().x() + step - (step > 0)));
//    }

//    template <int step>
//    __cuda_callable__ inline IndexType getEntityIndex() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0), "wrong coordinates");
//       TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates().x() + step - (step > 0) >= 0 &&
//                      entity.getCoordinates().x() + step - (step > 0) < entity.getMesh().getDimensions().x(),
//                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return this->entity.getIndex() + step - (step > 0);
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
// template <typename Real, typename Device, typename Index, typename Config>
// class NeighborGridEntityGetter<GridEntity<Meshes::Grid<1, Real, Device, Index>, 0>, 0> {
//   public:
//    static constexpr int EntityDimension = 0;
//    static constexpr int NeighborEntityDimension = 0;
//    typedef Meshes::Grid<1, Real, Device, Index> GridType;
//    typedef GridEntity<GridType, EntityDimension, Config> GridEntityType;
//    typedef GridEntity<GridType, NeighborEntityDimension, Config> NeighborGridEntityType;
//    typedef Real RealType;
//    typedef Index IndexType;
//    typedef typename GridType::CoordinatesType CoordinatesType;
//    typedef GridEntityGetter<GridType, NeighborGridEntityType> GridEntityGetterType;

//    __cuda_callable__ inline NeighborGridEntityGetter(const GridEntityType& entity) : entity(entity) {}

//    template <int step>
//    __cuda_callable__ inline NeighborGridEntityType getEntity() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0), "wrong coordinates");
//       TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates().x() + step >= CoordinatesType(0) && entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
//                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);
//       return NeighborGridEntity(CoordinatesType(entity.getCoordinates().x() + step));
//    }

//    template <int step>
//    __cuda_callable__ inline IndexType getEntityIndex() const {
//       TNL_ASSERT_GE(entity.getCoordinates(), CoordinatesType(0), "wrong coordinates");
//       TNL_ASSERT_LE(entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates");
//       TNL_ASSERT(entity.getCoordinates().x() + step >= CoordinatesType(0) && entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
//                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
//                            << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension);

//       return this->entity.getIndex() + step;
//    }

//    __cuda_callable__ void refresh(const GridType& grid, const IndexType& entityIndex){};

//   protected:
//    const GridEntityType& entity;
// };

// }  // namespace Meshes
// }  // namespace TNL
