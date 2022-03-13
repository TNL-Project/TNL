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

      EntityDataStore(const Index& entitiesCount) : entitiesCount(entitiesCount) {
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

      // NVCC is incapable of deducing generic lambda
      using UpdateFunctionType = std::function<void(const typename Grid::EntityType<EntityDimension>&)>;

      void storeAll(const Grid& grid, DataStore& store) const {
         this -> store(grid, store, [=] __cuda_callable__ (const UpdateFunctionType& update) { grid.template forAll<EntityDimension>(update); });
      }
      void storeBoundary(const Grid& grid, DataStore& store) const {
         this -> store(grid, store, [&](const UpdateFunctionType& update) { grid.template forBoundary<EntityDimension>(update); });
      }
      void storeInterior(const Grid& grid, DataStore& store) const {
         this -> store(grid, store, [&](const UpdateFunctionType& update) { grid.template forInterior<EntityDimension>(update); });
      }
      void clearAll(const Grid& grid, DataStore& store) const {
         clear(grid, store, [&](const UpdateFunctionType& update) { grid.template forAll<EntityDimension>(update); });
      }
      void clearBoundary(const Grid& grid, DataStore& store) const {
         clear(grid, store, [&](const UpdateFunctionType& update) { grid.template forBoundary<EntityDimension>(update); });
      }
      void clearInterior(const Grid& grid, DataStore& store) const {
         clear(grid, store, [&](const UpdateFunctionType& update) { grid.template forInterior<EntityDimension>(update); });
      }

      void verifyAll(const Grid& grid, const DataStore& store) const {
         auto hostStore = store.template move<TNL::Devices::Host>();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);

         ASSERT_GT(orientationsCount, 0) << "Every entity must have at least one orientation";

         auto callsView = hostStore.getCallsView();
         auto indicesView = hostStore.getIndicesView();
         auto isBoundaryView = hostStore.getIsBoundaryView();
         auto coordinatesView = hostStore.getCoordinatesView();
         auto basisView = hostStore.getBasisView();

         for (Index i = 0; i < callsView.getSize(); i++)
            EXPECT_EQ(callsView[i], 1) << "Expect each index to be called only once";

         // Test each traversion of each orientation
         auto gridDimension = grid.getMeshDimension();

         auto verify = [&](const auto orientation) {
            CoordinateIterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               EXPECT_EQ(callsView.getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            int i = 0;
            while (!iterator.next() && i++ <= 100) {
               auto index = iterator.getIndex();

               EXPECT_EQ(callsView[index], 1) << "Expect the index to be called once";
               EXPECT_EQ(indicesView[index], index) << "Expect the index was correctly set";

               auto coordinate = iterator.getCoordinate();
               auto basis = iterator.getBasis();

               for (Index i = 0; i < gridDimension; i++) {
                  EXPECT_EQ(coordinatesView[index * gridDimension + i], coordinate[i]) << "Expect the coordinates are the same on the same index";
                  EXPECT_EQ(basisView[index * gridDimension + i], basis[i]) << "Expect the coordinates are the same on the same index";
               }
            }
         };

         Templates::DescendingFor<orientationsCount - 1>::exec(verify);
      }

      void verifyBoundary(const Grid& grid, const DataStore& store) const {

      }
      void verifyInterior(const Grid& grid, const DataStore& store) const {

      }

      template<typename Traverser>
      void store(const Grid& grid, DataStore& store, const Traverser traverser) const {
         auto callsView = store.getCallsView();
         auto indicesView = store.getIndicesView();
         auto isBoundaryView = store.getIsBoundaryView();
         auto coordinatesView = store.getCoordinatesView();
         auto basisView = store.getBasisView();
         auto gridDimension = Grid::getMeshDimension();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            auto index = entity.getIndex();

            callsView[index] += 1;

            indicesView[index] = index;
            isBoundaryView[index] = entity.isBoundary();

            auto coordinates = entity.getCoordinates();

            for (Index i = 0; i < gridDimension; i++)
               coordinatesView[index * gridDimension + i] = coordinates[i];

            auto basis = entity.getBasis();

            for (Index i = 0; i < gridDimension; i++)
               basisView[index * gridDimension + i] = basis[i];
         };

         traverser(update);
      }

      template<typename Traverser>
      void clear(const Grid& grid, DataStore& store, Traverser traverser) const {
         auto callsView = store.getCallsView();
         auto indicesView = store.getIndicesView();
         auto isBoundaryView = store.getIsBoundaryView();
         auto coordinatesView = store.getCoordinatesView();
         auto basisView = store.getBasisView();
         auto gridDimension = Grid::getMeshDimension();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            auto index = entity.getIndex();

            callsView[index] = 0;
            indicesView[index] = 0;
            isBoundaryView[index] = 0;

            for (Index i = 0; i < gridDimension; i++)
               coordinatesView[index * gridDimension + i] = 0;

            for (Index i = 0; i < gridDimension; i++)
               basisView[index * gridDimension + i] = 0;
         };

         traverser(update);
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

            bool isBoundary() const {
               for (Index i = 0; i < current.getSize(); i++)
                  if (current[i] == 0 || current[i] == end[i])
                     return true;

               return false;
            }

            Coordinate getCoordinate() const {
               return current;
            }

            Index getIndex() const {
               Index result = 0;

               for (Index i = 0; i < current.getSize(); i++) {
                  if (i == 0) {
                     result += current[i];
                  } else {
                     Index offset = 0;

                     for (Index j = 0; j < i; j++)
                        offset += end[j];

                     result += current[i] * offset;
                  }
               }

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
};

template<typename Grid, int EntityDimension>
void testForAllTraverse(Grid& grid, const typename Grid::Coordinate& dimensions) {
   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeAll(grid, store);
   test.verifyAll(grid, store);
}

#endif
