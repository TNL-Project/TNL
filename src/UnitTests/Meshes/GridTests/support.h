#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#include <TNL/Containers/Array.h>

template<typename Device>
class GridTestCaseSupportInterface {
   public:
      template<typename Grid>
      void verifyDimensionGetters(const Grid& grid, const typename Grid::Coordinate& coordinates) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifyEntitiesCountGetters(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCount) const { FAIL() << "Expect to be specialized"; }


      template<typename Grid>
      void verifyOriginGetters(const Grid& grid, const typename Grid::Point& coordinates) const { FAIL() << "Expect to be specialized"; }


      template<typename Grid>
      void verifyDimensionSetByCoordinateGetter(const Grid& grid, const typename Grid::Coordinate& dimensions) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifyDimensionSetByIndexGetter(const Grid& grid, const typename Grid::Coordinate& dimensions) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifyDimensionSetByIndiciesGetter(const Grid& grid, const typename Grid::Coordinate& dimensions) const { FAIL() << "Expect to be specialized"; }


      template<typename Grid>
      void verifyEntitiesCountByContainerGetter(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, Grid::IndexType>& entitiesCounts) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifyEntitiesCountByIndexGetter(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, Grid::IndexType>& entitiesCounts) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifyEntitiesCountByIndiciesGetter(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, Grid::IndexType>& entitiesCounts) const { FAIL() << "Expect to be specialized"; }


      template<typename Grid, int EntityDimension>
      void verifyForAll(const Grid& grid) const { FAIL() << "Expect to be specialized"; }
};


template<typename Device>
class GridTestCaseSupport: public GridTestCaseSupportInterface<Device> {};

template<>
class GridTestCaseSupport<TNL::Devices::Host>: public GridTestCaseSupportInterface<TNL::Devices::Host> {
   public:
      template<typename Grid>
      void verifyDimensionGetters(const Grid& grid, const typename Grid::Coordinate& dimensions) const {
         this->verifyDimensionSetByCoordinateGetter<Grid>(grid, dimensions);
         this->verifyDimensionSetByIndexGetter<Grid>(grid, dimensions);
         this->verifyDimensionSetByIndiciesGetter<Grid>(grid, dimensions);
      }

      template<typename Grid>
      void verifyEntitiesCountGetters(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         this->verifyEntitiesCountByContainerGetter<Grid>(grid, entitiesCounts);
         this->verifyEntitiesCountByIndexGetter<Grid>(grid, entitiesCounts);
         this->verifyEntitiesCountByIndiciesGetter<Grid>(grid, entitiesCounts);
      }

      template<typename Grid>
      void verifyOriginGetter(const Grid& grid, const typename Grid::Point& coordinates) const {
         auto result = grid.getOrigin();

         EXPECT_EQ(coordinates, result) << "Verify, that the origin was correctly set";
      }

      template<typename Grid>
      void verifyDimensionSetByCoordinateGetter(const Grid& grid, const typename Grid::Coordinate& dimensions) const {
         auto result = grid.getDimensions();

         EXPECT_EQ(dimensions, result) << "Verify, that dimension container accessor returns valid dimension";
      }

      template<typename Grid>
      void verifyDimensionSetByIndexGetter(const Grid& grid, const typename Grid::Coordinate& dimensions) const {
         for (typename Grid::IndexType i = 0; i < dimensions.getSize(); i++)
            EXPECT_EQ(grid.getDimension(i), dimensions[i]) << "Verify, that index access is correct";
      }

      template<typename Grid>
      void verifyDimensionSetByIndiciesGetter(const Grid& grid, const typename Grid::Coordinate& dimensions) const {
         for (typename Grid::IndexType i = 0; i < dimensions.getSize(); i++) {
            auto repeatedDimensions = grid.getDimensions(i, i, i, i, i, i, i, i, i, i);

            EXPECT_EQ(repeatedDimensions.getSize(), 10) << "Verify, that all dimension indices are returned";

            for (typename Grid::IndexType j = 0; j < repeatedDimensions.getSize(); j++)
               EXPECT_EQ(repeatedDimensions[j], dimensions[i]) << "Verify, that it is possible to request the same dimension multiple times";
         }
      }

      template<typename Grid>
      void verifyEntitiesCountByContainerGetter(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         auto result = grid.getEntitiesCounts();

         EXPECT_EQ(entitiesCounts, result) << "Verify, that returns expected entities counts";
      }

      template<typename Grid>
      void verifyEntitiesCountByIndexGetter(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         for (typename Grid::IndexType i = 0; i < entitiesCounts.getSize(); i++)
            EXPECT_EQ(grid.getEntitiesCount(i), entitiesCounts[i]) << "Verify, that index access is correct";
      }

      template<typename Grid>
      void verifyEntitiesCountByIndiciesGetter(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         for (typename Grid::IndexType i = 0; i < entitiesCounts.getSize(); i++) {
            auto repeated = grid.getEntitiesCounts(i, i, i, i, i, i, i, i, i, i);

            EXPECT_EQ(repeated.getSize(), 10) << "Verify, that all dimension indices are returned";

            for (typename Grid::IndexType j = 0; j < repeated.getSize(); j++)
               EXPECT_EQ(repeated[j], entitiesCounts[i]) << "Verify, that it is possible to request the same dimension multiple times";
         }
      }

      template<typename Grid>
      void verifyForAll(const Grid& grid) const {



      }
};

template<>
class GridTestCaseSupport<TNL::Devices::Cuda>: public GridTestCaseSupportInterface<TNL::Devices::Cuda> {
   public:
      template<typename Grid>
      void verifyDimensionGetters(const Grid& grid, const typename Grid::Coordinate& dimensions) const {
         this->verifyDimensionSetByCoordinateGetter<Grid>(grid, dimensions);
         this->verifyDimensionSetByIndexGetter<Grid>(grid, dimensions);
         this->verifyDimensionSetByIndiciesGetter<Grid>(grid, dimensions);
      }

      template<typename Grid>
      void verifyEntitiesCountGetters(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCount) const {
         this->verifyEntitiesCountByContainerGetter<Grid>(grid, entitiesCount);
         this->verifyEntitiesCountByIndexGetter<Grid>(grid, entitiesCount);
         this->verifyEntitiesCountByIndiciesGetter<Grid>(grid, entitiesCount);
      }

      template<typename Grid>
      void verifyOriginGetter(const Grid& grid, const typename Grid::Point& coordinates) const {
         auto gridDimension = grid.getMeshDimension();

         auto update = [=] __device__ (const int index, typename Grid::RealType& reference) mutable {
            reference = grid.getOrigin()[index % gridDimension];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::RealType& reference) mutable {
            EXPECT_EQ(reference, coordinates[index % gridDimension]);
         };

         this->executeFromDevice<typename Grid::RealType>(update, verify);
      }


      template<typename ContainerElementType, typename Updater, typename Verifier>
      void executeFromDevice(Updater&& updater, Verifier&& verifier) const {
         TNL::Containers::Array<ContainerElementType, TNL::Devices::Cuda> container(100 * 100);

         container.forAllElements(updater);

         TNL::Containers::Array<ContainerElementType, TNL::Devices::Host> result(container);

         result.forAllElements(verifier);
      }


      template<typename Grid>
      void verifyDimensionSetByCoordinateGetter(const Grid& grid, const typename Grid::Coordinate& dimensions) const {
         auto gridDimension = grid.getMeshDimension();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getDimensions()[index % gridDimension];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, dimensions[index % gridDimension]);
         };

         this->executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid>
      void verifyDimensionSetByIndexGetter(const Grid& grid, const typename Grid::Coordinate& dimensions) const {
         auto gridDimension = grid.getMeshDimension();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getDimension(index % gridDimension);
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, dimensions[index % gridDimension]);
         };

         this->executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid>
      void verifyDimensionSetByIndiciesGetter(const Grid& grid, const typename Grid::Coordinate& dimensions) const {
         auto gridDimension = grid.getMeshDimension();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getDimensions(index % gridDimension)[0];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, dimensions[index % gridDimension]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }


      template<typename Grid>
      void verifyEntitiesCountByContainerGetter(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         auto size = entitiesCounts.getSize();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getEntitiesCounts()[index % size];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, entitiesCounts[index % size]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid>
      void verifyEntitiesCountByIndexGetter(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         auto size = entitiesCounts.getSize();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getEntitiesCount(index % size);
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, entitiesCounts[index % size]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid>
      void verifyEntitiesCountByIndiciesGetter(const Grid& grid, const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
          auto size = entitiesCounts.getSize();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getEntitiesCounts(index % size)[0];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, entitiesCounts[index % size]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }
};

template<typename... Parameters>
std::string makeString(Parameters... parameters) {
   std::ostringstream s;

   for (const auto& x: { parameters... })
      s << x << ", ";

   return s.str();
}

template<typename Grid, typename... T>
void testDimensionSetByIndex(Grid& grid, T... dimensions) {
   auto paramString = makeString(dimensions...);

   EXPECT_NO_THROW(grid.setDimensions(dimensions...)) << "Verify, that the set of" << paramString << " doesn't cause assert";

   GridTestCaseSupport<typename Grid::DeviceType> support;

   support.template verifyDimensionGetters<Grid>(grid, typename Grid::Coordinate(dimensions...));
}

template<typename Grid, typename... T>
void testDimensionSetByCoordinate(Grid& grid, const typename Grid::Coordinate& dimensions) {
   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   GridTestCaseSupport<typename Grid::DeviceType> support;

   support.template verifyDimensionGetters<Grid>(grid, dimensions);
}

template<typename Grid>
void testEntitiesCounts(Grid& grid,
                        const typename Grid::Coordinate& dimensions,
                        const typename Grid::Container<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) {
   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   GridTestCaseSupport<typename Grid::DeviceType> support;

   support.template verifyEntitiesCountGetters<Grid>(grid, entitiesCounts);
}

template<typename Grid,
         typename... T,
         std::enable_if_t<TNL::Meshes::Templates::conjunction<std::is_convertible<typename Grid::RealType, T>::value...>::value, bool> = true>
void testOriginSetByIndex(Grid& grid, T... coordinates) {
   auto paramString = makeString(coordinates...);

   EXPECT_NO_THROW(grid.setOrigin(coordinates...)) << "Verify, that the set of" << paramString << " doesn't cause assert";

   GridTestCaseSupport<typename Grid::DeviceType> support;

   support.template verifyOriginGetter<Grid>(grid, typename Grid::Point(coordinates...));
}

template<typename Grid>
void testOriginSetByCoordinate(Grid& grid, const typename Grid::Point& coordinates) {
   EXPECT_NO_THROW(grid.setOrigin(coordinates)) << "Verify, that the set of" << coordinates << " doesn't cause assert";

   GridTestCaseSupport<typename Grid::DeviceType> support;

   support.template verifyOriginGetter<Grid>(grid, coordinates);
}

#endif
