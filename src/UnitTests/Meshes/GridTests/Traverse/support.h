#pragma once

#ifdef HAVE_GTEST
#include <functional>
#include <TNL/Containers/Array.h>
#include <TNL/Meshes/Basis.h>
#include <gtest/gtest.h>

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

template<typename Index, typename Device, int GridDimension>
struct EntityDataStore {
   public:
      using Container = TNL::Containers::Array<Index, Device, Index>;

      struct View {
         public:
            View(typename Container::ViewType calls,
                 typename Container::ViewType indices,
                 typename Container::ViewType coordinates,
                 typename Container::ViewType isBoundary,
                 typename Container::ViewType basis)
               : calls(calls), indices(indices), coordinates(coordinates), isBoundary(isBoundary), basis(basis) {}

            template <typename Entity>
            __cuda_callable__ void store(const Entity entity) {
               auto index = entity.getIndex();

               calls[index] += 1;
               indices[index] = entity.getIndex();
               isBoundary[index] = entity.isBoundary();

               auto coordinates = entity.getCoordinates();
               auto basis = entity.getBasis();

               for (Index i = 0; i < GridDimension; i++) {
                  this -> coordinates[index * GridDimension + i] = coordinates[i];
                  this -> basis[index * GridDimension + i] = basis[i];
               }
            }

            template <typename Entity>
            __cuda_callable__ void clear(const Entity entity) {
               auto index = entity.getIndex();

               calls[index] = 0;
               indices[index] = 0;
               isBoundary[index] = 0;

               for (Index i = 0; i < GridDimension; i++) {
                  coordinates[index * GridDimension + i] = 0;
                  basis[index * GridDimension + i] = 0;
               }
            }
         private:
            typename Container::ViewType calls, indices, coordinates, basis, isBoundary;
      };

      EntityDataStore(const Index& entitiesCount)
          : entitiesCount(entitiesCount) {
         calls.resize(entitiesCount);
         indices.resize(entitiesCount);
         isBoundary.resize(entitiesCount);
         coordinates.resize(GridDimension * entitiesCount);
         basis.resize(GridDimension * entitiesCount);

         calls = 0;
         indices = 0;
         isBoundary = 0;
         coordinates = 0;
         basis = 0;
      }

      EntityDataStore(const Container& calls,
                      const Container& indices,
                      const Container& coordinates,
                      const Container& basis,
                      const Container& isBoundary)
          : calls(calls), indices(indices), coordinates(coordinates), basis(basis), isBoundary(isBoundary) {}

      template<typename NewDevice>
      EntityDataStore<Index, NewDevice, GridDimension> move() const {
         using NewContainer = TNL::Containers::Array<Index, NewDevice, Index>;

         EntityDataStore<Index, NewDevice, GridDimension> newContainer(NewContainer(this -> calls),
                                                                       NewContainer(this -> indices),
                                                                       NewContainer(this -> coordinates),
                                                                       NewContainer(this -> basis),
                                                                       NewContainer(this -> isBoundary));

         return newContainer;
      };

      View getView() { return { getCallsView(), getIndicesView(), getCoordinatesView(),  getIsBoundaryView(), getBasisView() }; }

      typename Container::ViewType getCallsView() { return calls.getView(); }
      typename Container::ViewType getIndicesView() { return indices.getView(); }
      typename Container::ViewType getIsBoundaryView() { return isBoundary.getView(); }
      typename Container::ViewType getCoordinatesView() { return coordinates.getView(); }
      typename Container::ViewType getBasisView() { return basis.getView(); }
   private:
      Index entitiesCount;

      Container calls, indices, coordinates, basis, isBoundary;
};



template<typename Grid, int EntityDimension>
class GridTraverseTestCase {
   public:
      using Index = typename Grid::IndexType;
      using Coordinate = typename Grid::Coordinate;
      using DataStore = EntityDataStore<Index, typename Grid::DeviceType, Grid::getMeshDimension()>;
      using HostDataStore = EntityDataStore<Index, TNL::Devices::Host, Grid::getMeshDimension()>;

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
            CoordinateIterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(callsView.getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            do {
               verifyEntity(grid, iterator, hostStore, true);
            } while (!iterator.next());
         };

         Templates::DescendingFor<orientationsCount - 1>::exec(verify);
      }
      void verifyBoundary(const Grid& grid, const DataStore& store) const {
         auto hostStore = store.template move<TNL::Devices::Host>();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);

         SCOPED_TRACE("Verifying forBoundary");
         SCOPED_TRACE("Orientations Count: " + TNL::convertToString(orientationsCount));

         ASSERT_GT(orientationsCount, 0) << "Every entity must have at least one orientation";

         // Test each traversion of each orientation
         auto gridDimension = grid.getMeshDimension();

         auto verify = [&](const auto orientation) {
            CoordinateIterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(hostStore.getCallsView().getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            do {
               verifyEntity(grid, iterator, hostStore, iterator.isBoundary(grid));
            } while (!iterator.next());
         };

         Templates::DescendingFor<orientationsCount - 1>::exec(verify);
      }
      void verifyInterior(const Grid& grid, const DataStore& store) const {
         auto hostStore = store.template move<TNL::Devices::Host>();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);

         SCOPED_TRACE("Verifying forInterior");
         SCOPED_TRACE("Orientations Count: " + TNL::convertToString(orientationsCount));

         ASSERT_GT(orientationsCount, 0) << "Every entity must have at least one orientation";

         // Test each traversion of each orientation
         auto gridDimension = grid.getMeshDimension();

         auto verify = [&](const auto orientation) {
            CoordinateIterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(hostStore.getCallsView().getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            do {
               verifyEntity(grid, iterator, hostStore, !iterator.isBoundary(grid));
            } while (!iterator.next());
         };

         Templates::DescendingFor<orientationsCount - 1>::exec(verify);
      }
   private:
      template<int Orientation>
      class CoordinateIterator {
         public:
            using EntityBasis = TNL::Meshes::Basis<Index, Orientation, EntityDimension, Grid::getMeshDimension()>;

            CoordinateIterator(const Coordinate& end): end(end + EntityBasis::getBasis()) {
               for (Index i = 0; i < current.getSize(); i++)
                  current[i] = 0;
            }

            bool isBoundary(const Grid& grid) const {
               switch (EntityDimension) {
               case Grid::getMeshDimension():
                  for (Index i = 0; i < current.getSize(); i++)
                     if (current[i] == 0 || current[i] == grid.getDimension(i) - 1)
                        return true;

                  break;
               default:
                  for (Index i = 0; i < current.getSize(); i++)
                     if (getBasis()[i] && (current[i] == 0 || current[i] == grid.getDimension(i)))
                        return true;
                  break;
               }

               return false;
            }

            Coordinate getCoordinate() const {
               return current;
            }

            Index getIndex(const Grid& grid) const {
               Index result = 0;

               for (Index i = 0; i < current.getSize(); i++) {
                  if (i == 0) {
                     result += current[i];
                  } else {
                     Index offset = 1;

                     for (Index j = 0; j < i; j++)
                        offset *= end[j];

                     result += current[i] * offset;
                  }
               }

               for (Index i = 0; i < Orientation; i++)
                  result += grid.getOrientedEntitiesCount(EntityDimension, i);

               return result;
            }

            Index getOrientedIndex(const Grid& grid, const Coordinate& orientation, Index index) const {
               if (EntityDimension == 0 || EntityDimension == grid.getMeshDimension())
                  return index;

               for (Index i = 0; i < orientation.getSize(); i++) {
                  if (orientation[i])
                     break;

                  index += grid.getOrientedEntitiesCount(EntityDimension, i);
               }

               return index;
            }

            bool next() {
               current[0] += 1;

               Index carry = 0;

               bool isEnded = false;

               for (Index i = 0; i < current.getSize(); i++) {
                  current[i] += carry;

                  if (current[i] == end[i]) {
                     carry = 1;
                     current[i] = 0;

                     isEnded = i == current.getSize() - 1;
                     continue;
                  }

                  break;
               }

               return isEnded;
            }

            bool canIterate() {
               for (Index i = 0; i < current.getSize(); i++)
                  if (current[i] >= end[i])
                     return false;

               return true;
            }


            Coordinate getBasis() const {
               return EntityBasis::getBasis();
            }
         private:
            Coordinate current, end;
      };

      template<int Orientation>
      void verifyEntity(const Grid& grid,
                        const CoordinateIterator<Orientation>& iterator,
                        HostDataStore& dataStore,
                        bool expectCall) const {
         auto gridDimension = grid.getMeshDimension();
         auto index = iterator.getIndex(grid);

         auto callsView = dataStore.getCallsView();
         auto indicesView = dataStore.getIndicesView();
         auto isBoundaryView = dataStore.getIsBoundaryView();

         EXPECT_EQ(callsView[index], expectCall ? 1 : 0) << "Expect the index to be called once. View [" << callsView << "]";
         EXPECT_EQ(indicesView[index], expectCall ? index : 0) << "Expect the index was correctly set. View [" << indicesView << "]";
         EXPECT_EQ(isBoundaryView[index], expectCall ? iterator.isBoundary(grid) : 0) << "Expect the index was correctly set. View [" << isBoundaryView << "]";

         auto coordinate = iterator.getCoordinate();
         auto basis = iterator.getBasis();

         auto coordinatesView = dataStore.getCoordinatesView();
         auto basisView = dataStore.getBasisView();

         Coordinate entityCoordinate;
         Coordinate entityBasis;

         for (Index i = 0; i < gridDimension; i++) {
            entityCoordinate[i] = coordinatesView[index * gridDimension + i];
            entityBasis[i] = basisView[index * gridDimension + i];
         }

         EXPECT_EQ(entityCoordinate, expectCall ? coordinate : Coordinate(0))
                << "Expect the coordinates are the same on the same index. "
                << "Entity Index: [" << index << "] "
                << "View [" << coordinatesView << "]";
         EXPECT_EQ(entityBasis, expectCall ? basis : Coordinate(0))
                << "Expect the bases are the same on the same index. "
                << "Entity Index: [" << index << "] "
                << "View [" << basisView << "]";
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
