#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#include <TNL/Containers/Array.h>

template<int...>
struct IntPack {};

template<typename Device>
class GridTestCaseSupportInterface {
   public:
      template<typename Grid, int... dimensions>
      void verifyDimensionGetters(const Grid& grid) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid, int ...entitiesCount>
      void verifyEntitiesCountGetters(const Grid& grid) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid,
               typename... T,
               std::enable_if_t<TNL::Meshes::Templates::conjunction<std::is_convertible<typename Grid::RealType, T>::value...>::value, bool> = true>
      void verifyOriginGetters(const Grid& grid, T... coordinates) const { FAIL() << "Expect to be specialized"; }
   protected:
      template<typename Grid, int... dimensions>
      void verifyDimensionSetByCoordinateGetter(const Grid& grid) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid, int... dimensions>
      void verifyDimensionSetByIndexGetter(const Grid& grid) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid, int... dimensions>
      void verifyDimensionSetByIndiciesGetter(const Grid& grid) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid, int... entitiesCounts>
      void verifyEntitiesCountByContainerGetter(const Grid& grid) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid, int... entitiesCounts>
      void verifyEntitiesCountByIndexGetter(const Grid& grid) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid, int... entitiesCounts>
      void verifyEntitiesCountByIndiciesGetter(const Grid& grid) const { FAIL() << "Expect to be specialized"; }
};


template<typename Device>
class GridTestCaseSupport: public GridTestCaseSupportInterface<Device> {};

template<>
class GridTestCaseSupport<TNL::Devices::Host>: public GridTestCaseSupportInterface<TNL::Devices::Host> {
   public:
      template<typename Grid, int... dimensions>
      void verifyDimensionGetters(const Grid& grid) const {
         this->verifyDimensionSetByCoordinateGetter<Grid, dimensions...>(grid);
         this->verifyDimensionSetByIndexGetter<Grid, dimensions...>(grid);
         this->verifyDimensionSetByIndiciesGetter<Grid, dimensions...>(grid);
      }

      template<typename Grid, int ...entitiesCount>
      void verifyEntitiesCountGetters(const Grid& grid) const {
         this->verifyEntitiesCountByContainerGetter<Grid, entitiesCount...>(grid);
         this->verifyEntitiesCountByIndexGetter<Grid, entitiesCount...>(grid);
         this->verifyEntitiesCountByIndiciesGetter<Grid, entitiesCount...>(grid);
      }

      template<typename Grid,
               typename... T,
               std::enable_if_t<TNL::Meshes::Templates::conjunction<std::is_convertible<typename Grid::RealType, T>::value...>::value, bool> = true>
      void verifyOriginGetter(const Grid& grid, T... coordinates) const {
         typename Grid::Point point(coordinates...);

         auto result = grid.getOrigin();

         EXPECT_EQ(point, result) << "Verify, that the origin was correctly set";
      }
   protected:
      template<typename Grid, int... dimensions>
      void verifyDimensionSetByCoordinateGetter(const Grid& grid) const {
         typename Grid::Coordinate coordinates(dimensions...);

         auto result = grid.getDimensions();

         EXPECT_EQ(coordinates, result) << "Verify, that dimension container accessor returns valid dimension";
      }

      template<typename Grid, int... dimensions>
      void verifyDimensionSetByIndexGetter(const Grid& grid) const {
         typename Grid::Coordinate coordinates(dimensions...);

         for (int i = 0; i < (int)coordinates.getSize(); i++) {
            EXPECT_EQ(grid.getDimension(i), coordinates[i]) << "Verify, that index access is correct";
         }
      }

      template<typename Grid, int... dimensions>
      void verifyDimensionSetByIndiciesGetter(const Grid& grid) const {
         typename Grid::Coordinate coordinates(dimensions...);

         for (int i = 0; i < (int)coordinates.getSize(); i++) {
            auto repeatedDimensions = grid.getDimensions(i, i, i, i, i, i, i, i, i, i);

            EXPECT_EQ(repeatedDimensions.getSize(), 10) << "Verify, that all dimension indices are returned";

            for (int j = 0; j < repeatedDimensions.getSize(); j++)
               EXPECT_EQ(repeatedDimensions[j], coordinates[i]) << "Verify, that it is possible to request the same dimension multiple times";
         }
      }

      template<typename Grid, int... entitiesCounts>
      void verifyEntitiesCountByContainerGetter(const Grid& grid) const {
         constexpr auto optionsSize = grid.getMeshDimension() + 1;
         typename Grid::Container<optionsSize, typename Grid::IndexType> counts(entitiesCounts...);

         auto result = grid.getEntitiesCounts();

         EXPECT_EQ(counts, result) << "Verify, that returns expected entities counts";
      }

      template<typename Grid, int... entitiesCounts>
      void verifyEntitiesCountByIndexGetter(const Grid& grid) const {
         constexpr auto optionsSize = grid.getMeshDimension() + 1;
         typename Grid::Container<optionsSize, typename Grid::IndexType> counts(entitiesCounts...);

         for (int i = 0; i < optionsSize; i++) {
            EXPECT_EQ(grid.getEntitiesCount(i), counts[i]) << "Verify, that index access is correct";
         }
      }

      template<typename Grid, int... entitiesCounts>
      void verifyEntitiesCountByIndiciesGetter(const Grid& grid) const {
         constexpr auto optionsSize = grid.getMeshDimension() + 1;
         typename Grid::Container<optionsSize, typename Grid::IndexType> counts(entitiesCounts...);

         for (int i = 0; i < optionsSize; i++) {
            auto repeated = grid.getEntitiesCounts(i, i, i, i, i, i, i, i, i, i);

            EXPECT_EQ(repeated.getSize(), 10) << "Verify, that all dimension indices are returned";

            for (int j = 0; j < repeated.getSize(); j++)
               EXPECT_EQ(repeated[j], counts[i]) << "Verify, that it is possible to request the same dimension multiple times";
         }
      }
};

template<>
class GridTestCaseSupport<TNL::Devices::Cuda>: public GridTestCaseSupportInterface<TNL::Devices::Cuda> {
   public:
      template<typename Grid, int... dimensions>
      void verifyDimensionGetters(const Grid& grid) const {
         this->verifyDimensionSetByCoordinateGetter<Grid, dimensions...>(grid);
         this->verifyDimensionSetByIndexGetter<Grid, dimensions...>(grid);
         this->verifyDimensionSetByIndiciesGetter<Grid, dimensions...>(grid);
      }

      template<typename Grid, int... entitiesCount>
      void verifyEntitiesCountGetters(const Grid& grid) const {
         this->verifyEntitiesCountByContainerGetter<Grid, entitiesCount...>(grid);
         this->verifyEntitiesCountByIndexGetter<Grid, entitiesCount...>(grid);
         this->verifyEntitiesCountByIndiciesGetter<Grid, entitiesCount...>(grid);
      }

      template<typename Grid,
               typename... T,
               std::enable_if_t<TNL::Meshes::Templates::conjunction<std::is_convertible<typename Grid::RealType, T>::value...>::value, bool> = true>
      void verifyOriginGetter(const Grid& grid, T... coordinates) const {
         auto gridDimension = grid.getMeshDimension();
         typename Grid::Point point(coordinates...);

         auto update = [=](const auto index, auto& reference) mutable {
            reference = grid.getOrigin()[index % gridDimension];
         };

         auto verify = [=](const auto index, const auto& reference) mutable {
            EXPECT_EQ(reference, point[index % gridDimension]);
         };

         this->executeFromDevice<typename Grid::IndexType>(update, verify);
      }
   protected:
      template<typename ContainerElementType, typename Updater, typename Verifier>
      void executeFromDevice(Updater&& updater, Verifier&& verifier) const {
         TNL::Containers::Array<ContainerElementType, TNL::Devices::Cuda> container(100 * 100);

         container.forAllElements(std::forward<Updater>(updater));

         TNL::Containers::Array<ContainerElementType, TNL::Devices::Host> result(container);

         result.forAllElements(std::forward<Verifier>(verifier));
      }

      template<typename Grid, int... dimensions>
      void verifyDimensionSetByCoordinateGetter(const Grid& grid) const {
         typename Grid::Coordinate coordinates(dimensions...);
         auto gridDimension = grid.getMeshDimension();

         auto update = [=](const auto index, auto& reference) mutable {
            reference = grid.getDimensions()[index % gridDimension];
         };

         auto verify = [=](const auto index, const auto& reference) mutable {
            EXPECT_EQ(reference, coordinates[index % gridDimension]);
         };

         this->executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid, int... dimensions>
      void verifyDimensionSetByIndexGetter(const Grid& grid) const {
         typename Grid::Coordinate coordinates(dimensions...);
         auto gridDimension = grid.getMeshDimension();

         auto update = [=](const auto index, auto& reference) mutable {
            reference = grid.getDimension(index % gridDimension);
         };

         auto verify = [=](const auto index, const auto& reference) mutable {
            EXPECT_EQ(reference, coordinates[index % gridDimension]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid, int... dimensions>
      void verifyDimensionSetByIndiciesGetter(const Grid& grid) const {
         typename Grid::Coordinate coordinates(dimensions...);
         auto gridDimension = grid.getMeshDimension();

         auto update = [=](const auto index, auto& reference) mutable {
            reference = grid.getDimensions(index % gridDimension)[0];
         };

         auto verify = [=](const auto index, const auto& reference) mutable {
            EXPECT_EQ(reference, coordinates[index % gridDimension]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid, int... entitiesCounts>
      void verifyEntitiesCountByContainerGetter(const Grid& grid) const {
         constexpr auto optionsSize = grid.getMeshDimension() + 1;
         typename Grid::Container<optionsSize, typename Grid::IndexType> counts(entitiesCounts...);

         auto update = [=](const auto index, auto& reference) mutable {
            reference = grid.getEntitiesCounts()[index % optionsSize];
         };

         auto verify = [=](const auto index, const auto& reference) mutable {
            EXPECT_EQ(reference, counts[index % optionsSize]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid, int... entitiesCounts>
      void verifyEntitiesCountByIndexGetter(const Grid& grid) const {
         constexpr auto optionsSize = grid.getMeshDimension() + 1;
         typename Grid::Container<optionsSize, typename Grid::IndexType> counts(entitiesCounts...);

         auto update = [=](const auto index, auto& reference) mutable {
            reference = grid.getEntitiesCount(index % optionsSize);
         };

         auto verify = [=](const auto index, const auto& reference) mutable {
            EXPECT_EQ(reference, counts[index % optionsSize]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid, int... entitiesCounts>
      void verifyEntitiesCountByIndiciesGetter(const Grid& grid) const {
         constexpr auto optionsSize = grid.getMeshDimension() + 1;
         typename Grid::Container<optionsSize, typename Grid::IndexType> counts(entitiesCounts...);

         auto update = [=](const auto index, auto& reference) mutable {
            reference = grid.getEntitiesCounts(index % optionsSize)[0];
         };

         auto verify = [=](const auto index, const auto& reference) mutable {
            EXPECT_EQ(reference, counts[index % optionsSize]);
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


template<typename Grid, bool isValid, int... dimensions>
void testDimensionSetByIndex(Grid& grid) {
   auto paramString = makeString(dimensions...);

   if (isValid) {
      EXPECT_NO_THROW(grid.setDimensions(dimensions...)) << "Verify, that the set of" << paramString << " doesn't cause assert";
   } else {
      EXPECT_ANY_THROW(grid.setDimensions(dimensions...)) << "Verify, that the set of " << paramString << " causes assert";
      return;
   }

   GridTestCaseSupport<typename Grid::DeviceType> support;

   support.template verifyDimensionGetters<Grid, dimensions...>(grid);
}

template<typename Grid, bool isValid, int... dimensions>
void testDimensionSetByCoordinate(Grid& grid) {
   auto paramString = makeString(dimensions...);
   typename Grid::Coordinate coordinate(dimensions...);

   if (isValid) {
      EXPECT_NO_THROW(grid.setDimensions(coordinate)) << "Verify, that the set of" << paramString << " doesn't cause assert";
   } else {
      EXPECT_ANY_THROW(grid.setDimensions(coordinate)) << "Verify, that the set of " << paramString << " causes assert";
      return;
   }

   GridTestCaseSupport<typename Grid::DeviceType> support;

   support.template verifyDimensionGetters<Grid, dimensions...>(grid);
}

template<typename Grid, typename, typename>
struct TestEntitiesCount;

template<typename Grid, int... dimensions, int... entitiesCounts>
struct TestEntitiesCount<Grid, IntPack<dimensions...>, IntPack<entitiesCounts...>> {
public:
   static void exec(Grid& grid) {
      auto paramString = makeString(dimensions...);

      EXPECT_NO_THROW(grid.setDimensions(dimensions...)) << "Verify, that the set of" << paramString << " doesn't cause assert";

      GridTestCaseSupport<typename Grid::DeviceType> support;

      support.template verifyEntitiesCountGetters<Grid, entitiesCounts...>(grid);
   }
};


template<typename Grid,
         typename... T,
         std::enable_if_t<TNL::Meshes::Templates::conjunction<std::is_convertible<typename Grid::RealType, T>::value...>::value, bool> = true>
void testOriginSetByIndex(Grid& grid, T... coordinates) {
   auto paramString = makeString(coordinates...);

   EXPECT_NO_THROW(grid.setOrigin(coordinates...)) << "Verify, that the set of" << paramString << " doesn't cause assert";

   GridTestCaseSupport<typename Grid::DeviceType> support;

   support.template verifyOriginGetter<Grid>(grid, coordinates...);
}

template<typename Grid,
         typename... T,
         std::enable_if_t<TNL::Meshes::Templates::conjunction<std::is_convertible<typename Grid::RealType, T>::value...>::value, bool> = true>
void testOriginSetByCoordinate(Grid& grid, T... coordinates) {
   auto paramString = makeString(coordinates...);
   typename Grid::Point point(coordinates...);

   EXPECT_NO_THROW(grid.setOrigin(point)) << "Verify, that the set of" << paramString << " doesn't cause assert";

   GridTestCaseSupport<typename Grid::DeviceType> support;

   support.template verifyOriginGetter<Grid>(grid, coordinates...);
}

#endif
