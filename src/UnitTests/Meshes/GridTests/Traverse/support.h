#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>
#include <functional>

#include <TNL/Containers/Array.h>
#include <TNL/Meshes/GridDetails/BasisGetter.h>

#include "../CoordinateIterator.h"
#include "../EntityDataStore.h"

namespace Templates {
/*
 * A compiler-friendly implementation of the templated for-cycle, because
 * the template specializations count is O(Value) bounded.
 */
template <int>
struct DescendingFor;

template <int Value>
struct DescendingFor {
  public:
   template <typename Func, typename... FuncArgs>
   static void exec(Func func, FuncArgs&&... args) {
      static_assert(Value > 0, "Couldn't descend for negative values");

      func(std::integral_constant<int, Value>(), std::forward<FuncArgs>(args)...);

      DescendingFor<Value - 1>::exec(std::forward<Func>(func), std::forward<FuncArgs>(args)...);
   }
};

template <>
struct DescendingFor<0> {
  public:
   template <typename Func, typename... FuncArgs>
   static void exec(Func func, FuncArgs&&... args) {
      func(std::integral_constant<int, 0>(), std::forward<FuncArgs>(args)...);
   }
};

}  // namespace Templates

template<typename Grid, int EntityDimension>
class GridTraverseTestCase {
   public:
      using Index = typename Grid::IndexType;
      using Real = typename Grid::RealType;
      using Coordinate = typename Grid::Coordinate;
      using DataStore = EntityDataStore<Index, Real, typename Grid::DeviceType, Grid::getMeshDimension()>;
      using HostDataStore = EntityDataStore<Index, Real, TNL::Devices::Host, Grid::getMeshDimension()>;

      // NVCC is incapable of deducing generic lambda
      using UpdateFunctionType = std::function<void(const typename Grid::EntityType<EntityDimension>&)>;

      void storeAll(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Store all");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            view.store(entity);
         };

         grid.template forAll<EntityDimension>(update);
      }
      void storeBoundary(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Store boundary");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            view.store(entity);
         };

         grid.template forBoundary<EntityDimension>(update);
      }
      void storeInterior(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Store interior");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            view.store(entity);
         };

         grid.template forInterior<EntityDimension>(update);
      }
      void clearAll(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Clear all");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            view.clear(entity);
         };

         grid.template forAll<EntityDimension>(update);
      }
      void clearBoundary(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Clear boundary");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            view.clear(entity);
         };

         grid.template forBoundary<EntityDimension>(update);
      }
      void clearInterior(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Clear interior");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            view.clear(entity);
         };

         grid.template forInterior<EntityDimension>(update);
      }

      void verifyAll(const Grid& grid, const DataStore& store) const {
         auto hostStore = store.template move<TNL::Devices::Host>();
         auto hostStoreView = hostStore.getView();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);

         SCOPED_TRACE("Verifying forAll");
         SCOPED_TRACE("Orientations Count: " + TNL::convertToString(orientationsCount));

         ASSERT_GT(orientationsCount, 0) << "Every entity must have at least one orientation";

         auto callsView = hostStore.getCallsView();

         for (Index i = 0; i < callsView.getSize(); i++)
            EXPECT_EQ(callsView[i], 1) << "Expect each index to be called only once";

         // Test each traversion of each orientation
         auto gridDimension = grid.getMeshDimension();

         auto verify = [&](const auto orientation) {
            GridCoordinateIterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(callsView.getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            do {
               verifyEntity(grid, iterator, hostStoreView, true);
            } while (!iterator.next());
         };

         Templates::DescendingFor<orientationsCount - 1>::exec(verify);
      }
      void verifyBoundary(const Grid& grid, const DataStore& store) const {
         auto hostStore = store.template move<TNL::Devices::Host>();
         auto hostStoreView = hostStore.getView();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);

         SCOPED_TRACE("Verifying forBoundary");
         SCOPED_TRACE("Orientations Count: " + TNL::convertToString(orientationsCount));

         ASSERT_GT(orientationsCount, 0) << "Every entity must have at least one orientation";

         // Test each traversion of each orientation
         auto gridDimension = grid.getMeshDimension();

         auto verify = [&](const auto orientation) {
            GridCoordinateIterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(hostStore.getCallsView().getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            do {
               verifyEntity(grid, iterator, hostStoreView, iterator.isBoundary(grid));
            } while (!iterator.next());
         };

         Templates::DescendingFor<orientationsCount - 1>::exec(verify);
      }
      void verifyInterior(const Grid& grid, const DataStore& store) const {
         auto hostStore = store.template move<TNL::Devices::Host>();
         auto hostStoreView = hostStore.getView();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);

         SCOPED_TRACE("Verifying forInterior");
         SCOPED_TRACE("Orientations Count: " + TNL::convertToString(orientationsCount));

         ASSERT_GT(orientationsCount, 0) << "Every entity must have at least one orientation";

         // Test each traversion of each orientation
         auto gridDimension = grid.getMeshDimension();

         auto verify = [&](const auto orientation) {
            GridCoordinateIterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(hostStore.getCallsView().getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            do {
               verifyEntity(grid, iterator, hostStoreView, !iterator.isBoundary(grid));
            } while (!iterator.next());
         };

         Templates::DescendingFor<orientationsCount - 1>::exec(verify);
      }
   private:
      template<int Orientation>
      class GridCoordinateIterator: public CoordinateIterator<typename Grid::IndexType, Grid::getMeshDimension()> {
         public:
            using Base = CoordinateIterator<typename Grid::IndexType, Grid::getMeshDimension()>;
            using EntityBasis = TNL::Meshes::BasisGetter<Index, EntityDimension, Grid::getMeshDimension()>;

            GridCoordinateIterator(const Coordinate& end): Base(Coordinate(0), end + EntityBasis::template getBasis<Orientation>()) {
               for (Index i = 0; i < this -> current.getSize(); i++) {
                  this -> start[i] = 0;
                  this -> current[i] = 0;
               }
            }

            bool isBoundary(const Grid& grid) const {
               switch (EntityDimension) {
               case Grid::getMeshDimension():
                  for (Index i = 0; i < this -> current.getSize(); i++)
                     if (this -> current[i] == 0 || this -> current[i] == grid.getDimension(i) - 1)
                        return true;

                  break;
               default:
                  for (Index i = 0; i < this -> current.getSize(); i++)
                     if (getBasis()[i] && (this -> current[i] == 0 || this -> current[i] == grid.getDimension(i)))
                        return true;
                  break;
               }

               return false;
            }

            Coordinate getCoordinate() const {
               return this -> current;
            }

            Index getIndex(const Grid& grid) const {
               Index result = 0;

               for (Index i = 0; i < this -> current.getSize(); i++) {
                  if (i == 0) {
                     result += this -> current[i];
                  } else {
                     Index offset = 1;

                     for (Index j = 0; j < i; j++)
                        offset *= this -> end[j];

                     result += this -> current[i] * offset;
                  }
               }

               for (Index i = 0; i < Orientation; i++)
                  result += grid.getOrientedEntitiesCount(EntityDimension, i);

               return result;
            }

            Coordinate getBasis() const {
               return EntityBasis::template getBasis<Orientation>();
            }
      };

      template<int Orientation>
      void verifyEntity(const Grid& grid,
                        const GridCoordinateIterator<Orientation>& iterator,
                        typename HostDataStore::View& dataStore,
                        bool expectCall) const {
         auto gridDimension = grid.getMeshDimension();
         auto index = iterator.getIndex(grid);

         auto entity = dataStore.getEntity(index);

         SCOPED_TRACE("Entity: " + TNL::convertToString(entity));

         EXPECT_EQ(entity.calls, expectCall ? 1 : 0) << "Expect the index to be called once";
         EXPECT_EQ(entity.index, expectCall ? index : 0) << "Expect the index was correctly set";
         EXPECT_EQ(entity.isBoundary, expectCall ? iterator.isBoundary(grid) : 0) << "Expect the index was correctly set" ;

         auto coordinate = iterator.getCoordinate();
         auto basis = iterator.getBasis();

         EXPECT_EQ(entity.coordinate, expectCall ? coordinate : Coordinate(0))
                << "Expect the coordinates are the same on the same index. ";
         EXPECT_EQ(entity.basis, expectCall ? basis : Coordinate(0))
                << "Expect the bases are the same on the same index. ";
      }
};

template<typename Grid, int EntityDimension>
void testForAllTraverse(Grid& grid, const typename Grid::Coordinate& dimensions) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeAll(grid, store);
   test.verifyAll(grid, store);
}

template<typename Grid, int EntityDimension>
void testForInteriorTraverse(Grid& grid, const typename Grid::Coordinate& dimensions) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeInterior(grid, store);
   test.verifyInterior(grid, store);
}

template<typename Grid, int EntityDimension>
void testForBoundaryTraverse(Grid& grid, const typename Grid::Coordinate& dimensions) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeBoundary(grid, store);
   test.verifyBoundary(grid, store);
}

template<typename Grid, int EntityDimension>
void testBoundaryUnionInteriorEqualAllProperty(Grid& grid, const typename Grid::Coordinate& dimensions) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeBoundary(grid, store);
   test.storeInterior(grid, store);
   test.verifyAll(grid, store);
}

template<typename Grid, int EntityDimension>
void testAllMinusBoundaryEqualInteriorProperty(Grid& grid, const typename Grid::Coordinate& dimensions) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeAll(grid, store);
   test.clearBoundary(grid, store);
   test.verifyInterior(grid, store);
}

template<typename Grid, int EntityDimension>
void testAllMinusInteriorEqualBoundaryProperty(Grid& grid, const typename Grid::Coordinate& dimensions) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeAll(grid, store);
   test.clearInterior(grid, store);
   test.verifyBoundary(grid, store);
}


#endif
