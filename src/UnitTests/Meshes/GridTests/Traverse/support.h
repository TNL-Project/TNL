#pragma once

#ifdef HAVE_GTEST
#include <functional>
#include <TNL/Containers/Array.h>
#include <gtest/gtest.h>

template<typename Index, typename Device, int GridDimension>
struct EntityDataStore {
   public:
      using Container = TNL::Containers::Array<Index, Device, Index>;

      EntityDataStore(const Index& entitiesCount) : entitiesCount(entitiesCount) {
         indices.resize(entitiesCount);
         isBoundary.resize(entitiesCount);
         coordinates.resize(GridDimension * entitiesCount);
         orientation.resize(GridDimension * entitiesCount);

         indices = 0;
         isBoundary = 0;
         coordinates = 0;
         orientation = 0;
      }

      template<typename NewDevice>
      EntityDataStore<Index, NewDevice, GridDimension> move() const {
         if (std::is_same<Device, NewDevice>::value)
            return *this;

         EntityDataStore<Index, NewDevice, GridDimension> newContainer(entitiesCount);

         newContainer.indices = indices;
         newContainer.coordinates = coordinates;
         newContainer.orientation = orientation;
         newContainer.isBoundary = isBoundary;

         return newContainer;
      };

      typename Container::ViewType getIndicesView() { return indices.getView(); }
      typename Container::ViewType getIsBoundaryView() { return isBoundary.getView(); }
      typename Container::ViewType getCoordinatesView() { return coordinates.getView(); }
      typename Container::ViewType getOrientationView() { return orientation.getView(); }
   private:
      Index entitiesCount;

      Container indices, coordinates, orientation, isBoundary;
};

template<typename Grid, int EntityDimension>
class GridTraverseTestCase {
   public:
      using Index = typename Grid::IndexType;
      using Coordinate = typename Grid::Coordinate;
      using DataStore = EntityDataStore<Index, typename Grid::DeviceType, Grid::getMeshDimension()>;

      // NVCC is incapable of deducing auto lambda input parameters
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

         CoordinateIterator iter(grid.getDimensions(), grid.getDimensions());

         EXPECT_EQ(0, 1) << hostStore.getIndicesView();
         EXPECT_EQ(0, 1) << hostStore.getCoordinatesView();
         EXPECT_EQ(0, 1) << hostStore.getIsBoundaryView();
         EXPECT_EQ(0, 1) << hostStore.getOrientationView();

         int i = 0;

         while (iter.hasNext() && i != 10) {
            EXPECT_EQ(0, 1) << iter.getIndex() << iter.getCoordinate();
            i++;

            iter.next();
         }
      }
      void verifyBoundary(const Grid& grid, const DataStore& store) const {

      }
      void verifyInterior(const Grid& grid, const DataStore& store) const {

      }


      template<typename Traverser>
      void store(const Grid& grid, DataStore& store, Traverser traverser) const {
         auto indicesView = store.getIndicesView();
         auto isBoundaryView = store.getIsBoundaryView();
         auto coordinatesView = store.getCoordinatesView();
         auto orientationView = store.getOrientationView();
         auto gridDimension = Grid::getMeshDimension();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            auto index = entity.getIndex();

            indicesView[index] = index;
            isBoundaryView[index] = entity.isBoundary();

            auto coordinates = entity.getCoordinates();

            for (Index i = 0; i < gridDimension; i++)
               coordinatesView[index * gridDimension + i] = coordinates[i];

            // auto orientation = entity.getOrientation();

            // for (Index i = 0; i < gridDimension; i++)
            //    orientationView[index * gridDimension + i] = orientation[i];
         };

         traverser(update);
      }

      template<typename Traverser>
      void clear(const Grid& grid, DataStore& store, Traverser traverser) const {
         auto indicesView = store.getIndicesView();
         auto isBoundaryView = store.getIsBoundaryView();
         auto coordinatesView = store.getCoordinatesView();
         auto orientationView = store.getOrientationView();
         auto gridDimension = Grid::getMeshDimension();

         auto update = [=] __cuda_callable__ (const typename Grid::EntityType<EntityDimension>& entity) mutable {
            auto index = entity.getIndex();

            indicesView[index] = 0;
            isBoundaryView[index] = 0;

            for (Index i = 0; i < gridDimension; i++)
               coordinatesView[index * gridDimension + i] = 0;

            for (Index i = 0; i < gridDimension; i++)
               orientationView[index * gridDimension + i] = 0;
         };

         traverser(update);
      }

   private:
      class CoordinateIterator {
         public:
            CoordinateIterator(const Coordinate& end, const Coordinate& orientation): end(end) {
               for (Index i = 0; i < current.getSize(); i++)
                  current[i] = 0;
            }

            bool isBoundary() const {
               for (Index i = 0; i < current.getSize(); i++)
                  if (current[i] == 0 || current[i] == end[i])
                     return true;

               return false;
            }

            Coordinate getCoordinate() {
               return current;
            }

            Index getIndex() {
               Index result = 0;

               for (Index i = 0; i < current.getSize(); i++)
                  result += current[i] * (i == 0 ? 1 : end[i]);

               return result;
            }

            Index getOrientedIndex(const Grid& grid, const Coordinate& orientation, Index index) {
               if (EntityDimension == 0 || EntityDimension == grid.getMeshDimension())
                  return index;

               for (Index i = 0; i < orientation.getSize(); i++) {
                  if (orientation[i])
                     break;

                  index += grid.getOrientedEntitiesCount(EntityDimension, i);
               }

               return index;
            }

            void next() {
               current[0] += 1;

               if (current == end)
                  return;

               Index carry = 0;

               for (Index i = 0; i < current.getSize(); i++) {
                  current[i] += carry;

                  if (current[i] == end[i] && i != current.getSize() - 1) {
                     carry = 1;
                     current[i] = 0;
                  } else {
                     break;
                  }
               }
            }

            bool hasNext() const {
               return current != end;
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
