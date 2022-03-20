
#pragma once

#include <TNL/Meshes/NDimGrid.h>

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
   Index i = 0;

   for (auto x : {dimensions...}) {
      TNL_ASSERT_GE(x, 0, "Dimension must be positive");
      this->dimensions[i] = x;
      i++;
   }

   fillEntitiesCount();
   fillSpaceSteps();
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::setDimensions(const typename __NDIM_PREFIX__::Coordinate &dimensions) {
   for (Index i = 0; i < Dimension; i++) {
      TNL_ASSERT_GE(dimensions[i], 0, "Dimension must be positive");
      this->dimensions[i] = dimensions[i];
   }

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

   for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++) result[i] = this->getDimension(result[i]);

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

   for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++) result[i] = this->cumulativeEntitiesCountAlongBases(result[i]);

   return result;
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline const typename __NDIM_PREFIX__::Container<Dimension + 1, Index> &__NDIM_PREFIX__::getEntitiesCounts() const noexcept {
   return this->cumulativeEntitiesCountAlongBases;
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::setOrigin(const typename __NDIM_PREFIX__::Point &origin) noexcept { this->origin = origin; }

__NDIMGRID_TEMPLATE__
template <typename... Coordinates,
          std::enable_if_t<Templates::conjunction_v<std::is_convertible<Real, Coordinates>...>, bool>,
          std::enable_if_t<sizeof...(Coordinates) == Dimension, bool>>
void __NDIM_PREFIX__::setOrigin(Coordinates... coordinates) noexcept {
   Index i = 0;

   for (auto x : {coordinates...}) {
      this->origin[i] = x;
      i++;
   }
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
template <typename... Coordinates, std::enable_if_t<Templates::conjunction_v<std::is_convertible<Real, Coordinates>...>, bool>,
          std::enable_if_t<sizeof...(Coordinates) == Dimension, bool>>
void __NDIM_PREFIX__::setSpaceSteps(Coordinates... coordinates) noexcept {
   Index i = 0;

   for (auto x : {coordinates...}) {
      this->spaceSteps[i] = x;
      i++;
   }

   fillSpaceStepsPowers();
   fillProportions();
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline const typename __NDIM_PREFIX__::Point &__NDIM_PREFIX__::getSpaceSteps() const noexcept { return this->spaceSteps; }

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline const typename __NDIM_PREFIX__::Point &__NDIM_PREFIX__::getProportions() const noexcept { return this->proportions; }

__NDIMGRID_TEMPLATE__
template <typename... Powers, std::enable_if_t<Templates::conjunction_v<std::is_convertible<Real, Powers>...>, bool>,
          std::enable_if_t<sizeof...(Powers) == Dimension, bool>>
__cuda_callable__ inline Real __NDIM_PREFIX__::getSpaceStepsProducts(Powers... powers) const noexcept {
   constexpr Index halfSize = this->spaceStepsPowersSize >> 1;

   Index i = 0;
   Index base = 1;

   for (auto x : {powers...}) {
      static_assert(x >= -halfSize && x <= halfSize, "Unsupported power");

      i += x * base;
      base *= this->spaceStepsPowersSize;
   }

   return this->spaceStepsProducts(i);
}

__NDIMGRID_TEMPLATE__
__cuda_callable__ inline Real __NDIM_PREFIX__::getSmallestSpaceSteps() const noexcept {
   Real minStep = this->spaceSteps[0];
   Index i = 1;

   while (i != Dimension) minStep = min(minStep, this->spaceSteps[i++]);

   return minStep;
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __NDIM_PREFIX__::traverseAll(Func func, FuncArgs... args) const {
   auto exec = [&](const Coordinate& basis) {
      Templates::ParallelFor<Dimension, Device, Index>::exec(Coordinate(0), this -> getDimensions() + basis, func, basis, args...);
   };

   ForEachOrientation<EntityDimension>::exec(exec);
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline
void __NDIM_PREFIX__::traverseInterior(Func func, FuncArgs... args) const {
   auto exec = [&](const Coordinate& basis) {
      switch (EntityDimension) {
      case 0: {
         Templates::ParallelFor<Dimension, Device, Index>::exec(Coordinate(1), this -> getDimensions(), func, basis, args...);
         break;
      }
      case Dimension: {
         Templates::ParallelFor<Dimension, Device, Index>::exec(Coordinate(1), this -> getDimensions() - Coordinate(1), func, basis, args...);
         break;
      }
      default: {
         Templates::ParallelFor<Dimension, Device, Index>::exec(basis, this -> getDimensions(), func, basis, args...);
         break;
      }
      }
   };

   ForEachOrientation<EntityDimension>::exec(exec);
}

__NDIMGRID_TEMPLATE__
template <int EntityDimension, typename Func, typename... FuncArgs>
inline void __NDIM_PREFIX__::traverseBoundary(Func func, FuncArgs... args) const {
   Coordinate from(0);
   Coordinate to = this->getDimensions();

   // Boundaries of the grid are formed by the entities of Dimension - 1.
   // We need to traverse each orientation independently.
   constexpr int orientationsCount = getEntityOrientationsCount(Dimension - 1);
   constexpr bool isDirectedEntity = EntityDimension != 0 && EntityDimension != Dimension;
   constexpr bool isAnyBoundaryIntersects = EntityDimension != Dimension - 1;

   Container<orientationsCount, Index> isBoundaryTraversed = { 0 };

   auto forBoundary = [&](const auto orientation, const Coordinate& basis) {
      Coordinate start = from;
      Coordinate end = to + basis;

      if (isAnyBoundaryIntersects) {
         #pragma unroll
         for (Index i = 0; i < Dimension; i++) {
            start[i] = (!isDirectedEntity || basis[i]) && isBoundaryTraversed[i] ? 1 : 0;
            end[i] = end[i] - ((!isDirectedEntity || basis[i]) && isBoundaryTraversed[i] ? 1 : 0);
         }
      }

      start[orientation] = end[orientation] - 1;

      Templates::ParallelFor<Dimension, Device, Index>::exec(start, end, func, basis, args...);

      // Skip entities defined only once
      if (!start[orientation] && end[orientation]) return;

      start[orientation] = 0;
      end[orientation] = 1;

      Templates::ParallelFor<Dimension, Device, Index>::exec(start, end, func, basis, args...);
   };

   if (!isAnyBoundaryIntersects) {
      auto exec = [&](const auto orientation, const Coordinate& basis) {
         constexpr int orthogonalOrientation = EntityDimension - orientation;

         forBoundary(orthogonalOrientation, basis);
      };

      ForEachOrientation<EntityDimension>::exec(exec);
      return;
   }

   auto exec = [&](const auto orthogonalOrientation) {
      auto exec = [&](const auto, const Coordinate& basis) {
         forBoundary(orthogonalOrientation, basis);
      };

      if (EntityDimension == 0 || EntityDimension == Dimension) {
         ForEachOrientation<EntityDimension>::exec(exec);
      } else {
         ForEachOrientation<EntityDimension, orthogonalOrientation>::exec(exec);
      }

      isBoundaryTraversed[orthogonalOrientation] = 1;
   };

   Templates::DescendingFor<orientationsCount - 1>::exec(exec);
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::writeProlog(Logger& logger) const noexcept {
   logger.writeParameter("Dimensions:", this->dimensions);

   logger.writeParameter("Origin:", this->origin);
   logger.writeParameter("Proportions:", this->proportions);
   logger.writeParameter("Space steps:", this->spaceSteps);

   for (Index i = 0; i <= Dimension; i++) {
      String tmp = String("Entities count along dimension ") + String(i) + ":";

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
      for (Index i = 0; i < Dimension; i++) this->spaceSteps[i] = this->proportions[i] / this->dimensions[i];

      fillSpaceStepsPowers();
   }
}

__NDIMGRID_TEMPLATE__
void __NDIM_PREFIX__::fillSpaceStepsPowers() {
   Container<spaceStepsPowersSize * Dimension, Real> powers;

   for (Index i = 0; i < Dimension; i++) {
      Index power = -2;

      for (Index j = 0; j < spaceStepsPowersSize; j++) {
         powers[i * spaceStepsPowersSize + j] = pow(this->spaceSteps[i], power);
         power++;
      }
   }

   for (Index i = 0; i < this->spaceProducts.getSize(); i++) {
      Real product = 1;
      Index index = i;

      for (Index j = 0; j < Dimension; j++) {
         Index residual = index % this->spaceStepsPowersSize;

         index /= this->spaceStepsPowersSize;

         product *= powers[j * spaceStepsPowersSize + residual];
      }

      spaceProducts[i] = product;
   }
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
