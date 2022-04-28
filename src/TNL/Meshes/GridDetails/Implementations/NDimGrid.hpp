
#pragma once

#include <TNL/Meshes/GridDetails/Templates/BooleanOperations.h>
#include <TNL/Meshes/GridDetails/BasisGetter.h>
#include <TNL/Meshes/GridDetails/Templates/Templates.h>
#include <TNL/Meshes/GridDetails/Templates/ParallelFor.h>
#include <TNL/Meshes/GridDetails/Templates/DescendingFor.h>
#include <TNL/Meshes/GridDetails/Templates/ForEachOrientation.h>

namespace TNL {
namespace Meshes {

#define __NDIMGRID_TEMPLATE__ template <int Dimension, typename Real, typename Device, typename Index>
#define __NDIM_PREFIX__ NDimGrid<Dimension, Real, Device, Index>

__NDIMGRID_TEMPLATE__
constexpr Index __NDIM_PREFIX__::getEntityOrientationsCount(const Index entityDimension) {
   return Templates::combination(entityDimension, Dimension);
}

__NDIMGRID_TEMPLATE__
template <typename... Dimensions, std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, Dimensions>...>, bool>,
          std::enable_if_t<sizeof...(Dimensions) == Dimension, bool>>
void __NDIM_PREFIX__::setDimensions(Dimensions... dimensions) {
   this->dimensions = Coordinate(dimensions...);

   TNL_ASSERT_GE(this -> dimensions, Coordinate(0), "Dimension must be positive");

   fillEntitiesCount();
   fillSpaceSteps();
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::setDimensions(const typename __NDIM_PREFIX__::Coordinate &dimensions) {
   this -> dimensions = dimensions;

   TNL_ASSERT_GE(this -> dimensions, Coordinate(0), "Dimension must be positive");

   fillEntitiesCount();
   fillSpaceSteps();
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline Index __NDIM_PREFIX__::getDimension(const Index index) const {
   TNL_ASSERT_GE(index, 0, "Index must be greater or equal to zero");
   TNL_ASSERT_LT(index, Dimension, "Index must be less than Dimension");

   return dimensions[index];
}

__NDIMGRID_TEMPLATE__
template <typename... DimensionIndex, std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, DimensionIndex>...>, bool>,
          std::enable_if_t<(sizeof...(DimensionIndex) > 0), bool>>
__cuda_callable__ inline __NDIM_PREFIX__::Container<sizeof...(DimensionIndex), Index> __NDIM_PREFIX__::getDimensions(
    DimensionIndex... indices) const noexcept {
   Container<sizeof...(DimensionIndex), Index> result{indices...};

   for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
      result[i] = this->getDimension(result[i]);

   return result;
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline const typename __NDIM_PREFIX__::Container<Dimension, Index> &__NDIM_PREFIX__::getDimensions() const noexcept {
   return this->dimensions;
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline Index __NDIM_PREFIX__::getEntitiesCount(const Index index) const {
   TNL_ASSERT_GE(index, 0, "Index must be greater than zero");
   TNL_ASSERT_LE(index, Dimension, "Index must be less than or equal to Dimension");

   return this->cumulativeEntitiesCountAlongBases(index);
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension,
          std::enable_if_t<Templates::isInClosedInterval(0, EntityDimension, Dimension), bool>>
__cuda_callable__ inline Index __NDIM_PREFIX__::getEntitiesCount() const noexcept {
   return this->cumulativeEntitiesCountAlongBases(EntityDimension);
}

__NDIMGRID_TEMPLATE__
template <typename... DimensionIndex,
          std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, DimensionIndex>...>, bool>,
          std::enable_if_t<(sizeof...(DimensionIndex) > 0), bool>>
__cuda_callable__ inline __NDIM_PREFIX__::Container<sizeof...(DimensionIndex), Index> __NDIM_PREFIX__::getEntitiesCounts(
    DimensionIndex... indices) const {
   Container<sizeof...(DimensionIndex), Index> result{indices...};

   for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
      result[i] = this->cumulativeEntitiesCountAlongBases(result[i]);

   return result;
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline const typename __NDIM_PREFIX__::EntitiesCounts& __NDIM_PREFIX__::getEntitiesCounts() const noexcept {
   return this->cumulativeEntitiesCountAlongBases;
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::setOrigin(const typename __NDIM_PREFIX__::Point &origin) noexcept { this->origin = origin; }

__NDIMGRID_TEMPLATE__
template <typename... Coordinates,
          std::enable_if_t<Templates::conjunction_v<std::is_convertible<Real, Coordinates>...>, bool>,
          std::enable_if_t<sizeof...(Coordinates) == Dimension, bool>>
void __NDIM_PREFIX__::setOrigin(Coordinates... coordinates) noexcept {
   this->origin = Point(coordinates...);
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline
Index __NDIM_PREFIX__::getOrientedEntitiesCount(const Index dimension, const Index orientation) const {
   TNL_ASSERT_GE(dimension, 0, "Dimension must be greater than zero");
   TNL_ASSERT_LE(dimension, Dimension, "Requested dimension must be less than or equal to Dimension");

   if (dimension == 0 || dimension == Dimension)
      return this -> getEntitiesCount(dimension);

   Index index = Templates::firstKCombinationSum(dimension, Dimension) + orientation;

   return this -> entitiesCountAlongBases[index];
}

__NDIMGRID_TEMPLATE__
template<int EntityDimension>
__cuda_callable__ inline
typename __NDIM_PREFIX__::Coordinate __NDIM_PREFIX__::getBasis(Index orientation) const noexcept {
   constexpr Index index = Templates::firstKCombinationSum(EntityDimension, Dimension);

   return this -> bases(index + orientation);
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension,
          int EntityOrientation,
          std::enable_if_t<Templates::isInClosedInterval(0, EntityDimension, Dimension), bool>,
          std::enable_if_t<Templates::isInClosedInterval(0, EntityOrientation, Dimension), bool>>
__cuda_callable__ inline
Index __NDIM_PREFIX__::getOrientedEntitiesCount() const noexcept {
   if (EntityDimension == 0 || EntityDimension == Dimension)
      return this -> getEntitiesCount(EntityDimension);

   constexpr Index index = Templates::firstKCombinationSum(EntityDimension, Dimension) + EntityOrientation;

   return this -> entitiesCountAlongBases[index];
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline const typename __NDIM_PREFIX__::Point &__NDIM_PREFIX__::getOrigin() const noexcept { return this->origin; }

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::setDomain(const typename __NDIM_PREFIX__::Point &origin, const typename __NDIM_PREFIX__::Point &proportions) {
   this->origin = origin;
   this->proportions = proportions;

   this->fillSpaceSteps();
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::setSpaceSteps(const typename __NDIM_PREFIX__::Point &spaceSteps) noexcept {
   this->spaceSteps = spaceSteps;

   fillSpaceStepsPowers();
   fillProportions();
}

__NDIMGRID_TEMPLATE__
template <typename... Steps,
          std::enable_if_t<Templates::conjunction_v<std::is_convertible<Real, Steps>...>, bool>,
          std::enable_if_t<sizeof...(Steps) == Dimension, bool>>
void __NDIM_PREFIX__::setSpaceSteps(Steps... spaceSteps) noexcept {
   this->spaceSteps = Point(spaceSteps...);

   fillSpaceStepsPowers();
   fillProportions();
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline const typename __NDIM_PREFIX__::Point &__NDIM_PREFIX__::getSpaceSteps() const noexcept { return this->spaceSteps; }

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline const typename __NDIM_PREFIX__::Point &__NDIM_PREFIX__::getProportions() const noexcept { return this->proportions; }

__NDIMGRID_TEMPLATE__
template <typename... Powers,
          std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, Powers>...>, bool>,
          std::enable_if_t<sizeof...(Powers) == Dimension, bool>>
__cuda_callable__ inline
Real __NDIM_PREFIX__::getSpaceStepsProducts(Powers... powers) const {
   int index = Templates::makeCollapsedIndex(this -> spaceStepsPowersSize, Coordinate(powers...));

   return this->spaceStepsProducts(index);
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline Real __NDIM_PREFIX__::getSpaceStepsProducts(const Coordinate& powers) const {
   int index = Templates::makeCollapsedIndex(this -> spaceStepsPowersSize, powers);

   return this->spaceStepsProducts(index);
}

__NDIMGRID_TEMPLATE__
template <Index... Powers,
          std::enable_if_t<sizeof...(Powers) == Dimension, bool>>
__cuda_callable__ inline
Real __NDIM_PREFIX__::getSpaceStepsProducts() const noexcept {
   constexpr int index = Templates::makeCollapsedIndex<Index, Powers...>(this -> spaceStepsPowersSize);

   return this -> spaceStepsProducts(index);
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline Real __NDIM_PREFIX__::getSmallestSpaceStep() const noexcept {
   Real minStep = this->spaceSteps[0];
   Index i = 1;

   while (i != Dimension)
      minStep = min(minStep, this->spaceSteps[i++]);

   return minStep;
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __NDIM_PREFIX__::traverseAll(Func func, FuncArgs... args) const {
   this -> traverseAll<EntityDimension>(Coordinate(0),
                                       this -> getDimensions(),
                                       func,
                                       args...);
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __NDIM_PREFIX__::traverseAll(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const {
   TNL_ASSERT_GE(from, Coordinate(0), "Traverse rect must be in the grid dimensions");
   TNL_ASSERT_LE(to, this -> getDimensions(), "Traverse rect be in the grid dimensions");
   TNL_ASSERT_LE(from, to, "Traverse rect must be defined from leading bottom anchor to trailing top anchor");

   auto exec = [&](const Index orientation, const Coordinate& basis) {
      Templates::ParallelFor<Dimension, Device, Index>::exec(from,
                                                             to + basis,
                                                             func,
                                                             basis,
                                                             orientation,
                                                             args...);
   };

   Templates::ForEachOrientation<Index, EntityDimension, Dimension>::exec(exec);
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __NDIM_PREFIX__::traverseInterior(Func func, FuncArgs... args) const {
   this -> traverseInterior<EntityDimension>(Coordinate(0),
                                             this -> getDimensions(),
                                             func,
                                             args...);
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __NDIM_PREFIX__::traverseInterior(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const {
   TNL_ASSERT_GE(from, Coordinate(0), "Traverse rect must be in the grid dimensions");
   TNL_ASSERT_LE(to, this -> getDimensions(), "Traverse rect be in the grid dimensions");
   TNL_ASSERT_LE(from, to, "Traverse rect must be defined from leading bottom anchor to trailing top anchor");

   auto exec = [&](const Index orientation, const Coordinate& basis) {
      switch (EntityDimension) {
      case 0: {
         Templates::ParallelFor<Dimension, Device, Index>::exec(from + Coordinate(1),
                                                                to,
                                                                func,
                                                                basis,
                                                                orientation,
                                                                args...);
         break;
      }
      case Dimension: {
         Templates::ParallelFor<Dimension, Device, Index>::exec(from + Coordinate(1),
                                                                to - Coordinate(1),
                                                                func,
                                                                basis,
                                                                orientation,
                                                                args...);
         break;
      }
      default: {
         Templates::ParallelFor<Dimension, Device, Index>::exec(from + basis,
                                                                to,
                                                                func,
                                                                basis,
                                                                orientation,
                                                                args...);
         break;
      }
      }
   };

   Templates::ForEachOrientation<Index, EntityDimension, Dimension>::exec(exec);
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
 void __NDIM_PREFIX__::traverseBoundary(Func func, FuncArgs... args) const {
   this -> traverseBoundary<EntityDimension>(Coordinate(0),
                                             this -> getDimensions(),
                                             func,
                                             args...);
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __NDIM_PREFIX__::traverseBoundary(const Coordinate& from, const Coordinate& to, Func func, FuncArgs... args) const {
   // Boundaries of the grid are formed by the entities of Dimension - 1.
   // We need to traverse each orientation independently.
   constexpr int orientationsCount = getEntityOrientationsCount(Dimension - 1);
   constexpr bool isDirectedEntity = EntityDimension != 0 && EntityDimension != Dimension;
   constexpr bool isAnyBoundaryIntersects = EntityDimension != Dimension - 1;

   Container<orientationsCount, Index> isBoundaryTraversed(0);

   auto forBoundary = [&](const auto orthogonalOrientation, const auto orientation, const Coordinate& basis) {
      Coordinate start = from;
      Coordinate end = to + basis;

      if (isAnyBoundaryIntersects) {
         #pragma unroll
         for (Index i = 0; i < Dimension; i++) {
            start[i] = (!isDirectedEntity || basis[i]) && isBoundaryTraversed[i] ? 1 : 0;
            end[i] = end[i] - ((!isDirectedEntity || basis[i]) && isBoundaryTraversed[i] ? 1 : 0);
         }
      }

      start[orthogonalOrientation] = end[orthogonalOrientation] - 1;

      Templates::ParallelFor<Dimension, Device, Index>::exec(start,
                                                             end,
                                                             func,
                                                             basis,
                                                             orientation,
                                                             args...);

      // Skip entities defined only once
      if (!start[orthogonalOrientation] && end[orthogonalOrientation]) return;

      start[orthogonalOrientation] = 0;
      end[orthogonalOrientation] = 1;

      Templates::ParallelFor<Dimension, Device, Index>::exec(start,
                                                             end,
                                                             func,
                                                             basis,
                                                             orientation,
                                                             args...);
   };

   if (!isAnyBoundaryIntersects) {
      auto exec = [&](const auto orientation, const Coordinate& basis) {
         constexpr int orthogonalOrientation = EntityDimension - orientation;

         forBoundary(orthogonalOrientation, orientation, basis);
      };

      Templates::ForEachOrientation<Index, EntityDimension, Dimension>::exec(exec);
      return;
   }

   auto exec = [&](const auto orthogonalOrientation) {
      auto exec = [&](const auto orientation, const Coordinate& basis) {
         forBoundary(orthogonalOrientation, orientation, basis);
      };

      if (EntityDimension == 0 || EntityDimension == Dimension) {
         Templates::ForEachOrientation<Index, EntityDimension, Dimension>::exec(exec);
      } else {
         Templates::ForEachOrientation<Index, EntityDimension, Dimension, orthogonalOrientation>::exec(exec);
      }

      isBoundaryTraversed[orthogonalOrientation] = 1;
   };

   Templates::DescendingFor<orientationsCount - 1>::exec(exec);
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::writeProlog(TNL::Logger& logger) const noexcept {
   logger.writeParameter("Dimensions:", this->dimensions);

   logger.writeParameter("Origin:", this->origin);
   logger.writeParameter("Proportions:", this->proportions);
   logger.writeParameter("Space steps:", this->spaceSteps);

   for (Index i = 0; i <= Dimension; i++) {
      TNL::String tmp = TNL::String("Entities count along dimension ") + TNL::convertToString(i) + ":";

      logger.writeParameter(tmp, this->cumulativeEntitiesCountAlongBases[i]);
   }
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::fillEntitiesCount() {
   Index j = 0;

   for (Index i = 0; i < Dimension + 1; i++) cumulativeEntitiesCountAlongBases[i] = 0;

   // In case, if some dimension is zero. Clear all counts
   for (Index i = 0; i < Dimension; i++) {
      if (dimensions[i] == 0) {
         for (Index k = 0; k < (Index)entitiesCountAlongBases.getSize(); k++)
            entitiesCountAlongBases[k] = 0;

         return;
      }
   }

   for (Index i = 0; i <= Dimension; i++) {
      forEachPermutation(Dimension - i, Dimension, [&](const std::vector<Index>& permutation) {
         int result = 1;

         for (Index k = 0; k < (Index)permutation.size(); k++)
            result *= dimensions[k] + permutation[k];

         entitiesCountAlongBases[j] = result;
         cumulativeEntitiesCountAlongBases[i] += result;

         j++;
      });
   }
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::fillProportions() {
   Index i = 0;

   while (i != Dimension) {
      this->proportions[i] = this->spaceSteps[i] * this->dimensions[i];
      i++;
   }
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::fillSpaceSteps() {
   bool hasAnyInvalidDimension = false;

   for (Index i = 0; i < Dimension; i++) {
      if (this->dimensions[i] <= 0) {
         hasAnyInvalidDimension = true;
         break;
      }
   }

   if (!hasAnyInvalidDimension) {
      for (Index i = 0; i < Dimension; i++)
         this->spaceSteps[i] = this->proportions[i] / this->dimensions[i];

      fillSpaceStepsPowers();
   }
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::fillSpaceStepsPowers() {
   Container<spaceStepsPowersSize * Dimension, Real> powers;

   for (Index i = 0; i < Dimension; i++) {
      Index power = -(this -> spaceStepsPowersSize >> 1);

      for (Index j = 0; j < spaceStepsPowersSize; j++) {
         powers[i * spaceStepsPowersSize + j] = pow(this->spaceSteps[i], power);
         power++;
      }
   }

   for (Index i = 0; i < this->spaceStepsProducts.getSize(); i++) {
      Real product = 1;
      Index index = i;

      for (Index j = 0; j < Dimension; j++) {
         Index residual = index % this->spaceStepsPowersSize;

         index /= this->spaceStepsPowersSize;

         product *= powers[j * spaceStepsPowersSize + residual];
      }

      spaceStepsProducts[i] = product;
   }
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::fillBases() {
   OrientationBasesContainer container;

   int index = container.getSize() - 1;

   auto forEachEntityDimension = [&](const auto entityDimension) {
      constexpr Index combinationsCount = this -> getEntityOrientationsCount(entityDimension);

      auto forEachOrientation = [&](const auto orientation, const auto entityDimension) {
         container[index--] = BasisGetter<Index, entityDimension, Dimension>::template getBasis<orientation>();
      };

      Templates::DescendingFor<combinationsCount - 1>::exec(forEachOrientation, entityDimension);
   };

   Templates::DescendingFor<Dimension>::exec(forEachEntityDimension);

   this -> bases = container;
}

__NDIMGRID_TEMPLATE__
template <typename Func, typename... FuncArgs>
void __NDIM_PREFIX__::forEachPermutation(const Index k, const Index n, Func func, FuncArgs... args) const {
   std::vector<int> buffer = {};

   buffer.resize(n);

   std::fill(buffer.begin(), buffer.end(), 0);
   std::fill(buffer.end() - k, buffer.end(), 1);

   do {
      func(buffer, args...);
   } while (std::next_permutation(buffer.begin(), buffer.end()));
}

}  // namespace Meshes
}  // namespace TNL
