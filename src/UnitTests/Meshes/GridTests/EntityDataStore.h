
#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticVector.h>

template <typename Index, typename Real, int GridDimension>
struct EntityPrototype {
  public:
   using Coordinate = TNL::Containers::StaticVector<GridDimension, Index>;

   EntityPrototype(const Coordinate& coordinate,
                   const Coordinate& basis,
                   const Index index,
                   const Index calls,
                   const Index orientation,
                   const bool isBoundary): coordinate(coordinate), basis(basis), index(index), calls(calls), orientation(orientation), isBoundary(isBoundary) {}

   const Coordinate coordinate;
   const Coordinate basis;
   const Index index;
   const Index calls;
   const Index orientation;
   const bool isBoundary;

   template<typename EntityIndex, typename EntityReal, int EntityGridDimension>
   friend std::ostream & operator << (std::ostream & os, const EntityPrototype<EntityIndex, EntityReal, EntityGridDimension>& entity);
};

template<typename EntityIndex, typename EntityReal, int EntityGridDimension>
std::ostream & operator << (std::ostream & os, const EntityPrototype<EntityIndex, EntityReal, EntityGridDimension>& entity) {
   os << "Coordinate: " << entity.coordinate << std::endl;
   os << "Basis: " << entity.basis << std::endl;
   os << "Index: " << entity.index << std::endl;
   os << "Calls: " << entity.calls << std::endl;
   os << "Orientation: " << entity.orientation << std::endl;
   os << "Is boundary: " << entity.isBoundary << std::endl;

   return os;
}

template<typename Index, typename Real, typename Device, int GridDimension>
struct EntityDataStore {
   public:
      using Coordinate = TNL::Containers::StaticVector<GridDimension, Index>;

      template<typename Value>
      using Container = TNL::Containers::Array<Value, Device, Index>;

      struct View {
         View(typename Container<Index>::ViewType calls,
              typename Container<Index>::ViewType indices,
              typename Container<Index>::ViewType coordinates,
              typename Container<Index>::ViewType basis,
              typename Container<Index>::ViewType orientations,
              typename Container<Index>::ViewType isBoundary): calls(calls), indices(indices), coordinates(coordinates), basis(basis), orientations(orientations), isBoundary(isBoundary) {}

         template <typename Entity>
         __cuda_callable__ void store(const Entity entity) {
            this -> store(entity, entity.getIndex());
         }

         template <typename Entity>
         __cuda_callable__ void store(const Entity entity, const Index index) {
            calls[index] += 1;
            indices[index] = entity.getIndex();
            isBoundary[index] = entity.isBoundary();
            orientations[index] = entity.getOrientation();

            auto coordinates = entity.getCoordinates();
            auto basis = entity.getBasis();

            for (Index i = 0; i < GridDimension; i++) {
               this->coordinates[index * GridDimension + i] = coordinates[i];
               this->basis[index * GridDimension + i] = basis[i];
            }
         }

         template <typename Entity>
         __cuda_callable__ void clear(const Entity entity) {
            auto index = entity.getIndex();

            calls[index] = 0;
            indices[index] = 0;
            isBoundary[index] = 0;
            orientations[index] = 0;

            for (Index i = 0; i < GridDimension; i++) {
               coordinates[index * GridDimension + i] = 0;
               basis[index * GridDimension + i] = 0;
            }
         }

         EntityPrototype<Index, Real, GridDimension> getEntity(const Index index) {
            Coordinate coordinates, basis;

            for (Index i = 0; i < GridDimension; i++) {
               coordinates[i] = this -> coordinates[index * GridDimension + i];
               basis[i] = this -> basis[index * GridDimension + i];
            }

            return { coordinates, basis, indices[index], calls[index], orientations[index], isBoundary[index] > 0 };
         }

         protected:
            typename Container<Index>::ViewType calls, indices, coordinates, basis, orientations, isBoundary;
      };

      EntityDataStore(const Index& entitiesCount): entitiesCount(entitiesCount) {
         calls.resize(entitiesCount);
         indices.resize(entitiesCount);
         isBoundary.resize(entitiesCount);
         orientations.resize(entitiesCount);
         coordinates.resize(GridDimension * entitiesCount);
         basis.resize(GridDimension * entitiesCount);

         calls = 0;
         indices = 0;
         isBoundary = 0;
         orientations = 0;
         coordinates = 0;
         basis = 0;
      }

      EntityDataStore(const Index& entitiesCount,
                      const Container<Index>& calls,
                      const Container<Index>& indices,
                      const Container<Index>& coordinates,
                      const Container<Index>& basis,
                      const Container<Index>& orientations,
                      const Container<Index>& isBoundary)
          : entitiesCount(entitiesCount),
            calls(calls),
            indices(indices),
            coordinates(coordinates),
            orientations(orientations),
            basis(basis),
            isBoundary(isBoundary) {}

      View getView() {
         return { calls.getView(), indices.getView(), coordinates.getView(), basis.getView(), orientations.getView(), isBoundary.getView() };
      }

      template<typename NewDevice>
      EntityDataStore<Index, Real, NewDevice, GridDimension> move() const {
         using NewIndexContainer = TNL::Containers::Array<Index, NewDevice, Index>;

         EntityDataStore<Index, Real, NewDevice, GridDimension> newContainer(this -> entitiesCount,
                                                                             NewIndexContainer(this -> calls),
                                                                             NewIndexContainer(this -> indices),
                                                                             NewIndexContainer(this -> coordinates),
                                                                             NewIndexContainer(this -> basis),
                                                                             NewIndexContainer(this -> orientations),
                                                                             NewIndexContainer(this -> isBoundary));

         return newContainer;
      };

      typename Container<Index>::ViewType getCallsView() { return calls.getView(); }
   private:
      Index entitiesCount;

      Container<Index> calls, indices, coordinates, orientations, basis, isBoundary;
};
