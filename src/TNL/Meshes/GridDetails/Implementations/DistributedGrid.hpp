
#pragma once

#include <TNL/Meshes/DistributedGrid.h>

namespace TNL {
namespace Meshes {
// TODO: Add checks

#define __DISTRIBUTED_GRID_TEMPLATE__ template <int Dimension, typename Real, typename Device, typename Index>
#define __DISTRIBUTED_GRID_PREFIX__ _DistributedGrid<Dimension, Real, Device, Index>

__DISTRIBUTED_GRID_TEMPLATE__
template <typename... Coordinates,
          std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, Coordinates>...>, bool>,
          std::enable_if_t<sizeof...(Coordinates) == Dimension, bool>>
void __DISTRIBUTED_GRID_PREFIX__::setLocalBegin(Coordinates... coordinates) noexcept {
   Index i = 0;

   for (auto x : {coordinates...}) {
      this->localBegin[i] = x;
      i++;
   }
}

__DISTRIBUTED_GRID_TEMPLATE__
void __DISTRIBUTED_GRID_PREFIX__::setLocalBegin(const Container<Dimension, Index> &coordinates) noexcept {
   this->localBegin = coordinates;
}

__DISTRIBUTED_GRID_TEMPLATE__
__cuda_callable__ __DISTRIBUTED_GRID_PREFIX__::Container<Dimension, Index>
__DISTRIBUTED_GRID_PREFIX__::getLocalBegin() const {
   return this->localBegin;
}

__DISTRIBUTED_GRID_TEMPLATE__
template <typename... Coordinates,
          std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, Coordinates>...>, bool>,
          std::enable_if_t<sizeof...(Coordinates) == Dimension, bool>>
void __DISTRIBUTED_GRID_PREFIX__::setLocalEnd(Coordinates... coordinates) noexcept {
   Index i = 0;

   for (auto x : {coordinates...}) {
      this->localEnd[i] = x;
      i++;
   }
}

__DISTRIBUTED_GRID_TEMPLATE__
void __DISTRIBUTED_GRID_PREFIX__::setLocalEnd(const Container<Dimension, Index> &coordinates) noexcept {
   this->localEnd = coordinates;
}

__DISTRIBUTED_GRID_TEMPLATE__
__cuda_callable__ __DISTRIBUTED_GRID_PREFIX__::Container<Dimension, Index>
__DISTRIBUTED_GRID_PREFIX__::getLocalEnd() const {
   return this->localEnd;
}

__DISTRIBUTED_GRID_TEMPLATE__
template <typename... Coordinates,
          std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, Coordinates>...>, bool>,
          std::enable_if_t<sizeof...(Coordinates) == Dimension, bool>>
void __DISTRIBUTED_GRID_PREFIX__::setInteriorBegin(Coordinates... coordinates) noexcept {
   Index i = 0;

   for (auto x : {coordinates...}) {
      this->interiorBegin[i] = x;
      i++;
   }
}

__DISTRIBUTED_GRID_TEMPLATE__
void __DISTRIBUTED_GRID_PREFIX__::setInteriorBegin(const Container<Dimension, Index> &coordinates) noexcept {
   this->interiorBegin = coordinates;
}

__DISTRIBUTED_GRID_TEMPLATE__
__cuda_callable__ __DISTRIBUTED_GRID_PREFIX__::Container<Dimension, Index>
__DISTRIBUTED_GRID_PREFIX__::getInteriorBegin() const {
   return this->interiorBegin;
}

__DISTRIBUTED_GRID_TEMPLATE__
template <typename... Coordinates,
          std::enable_if_t<Templates::conjunction_v<std::is_convertible<Index, Coordinates>...>, bool>,
          std::enable_if_t<sizeof...(Coordinates) == Dimension, bool>>
void __DISTRIBUTED_GRID_PREFIX__::setInteriorEnd(Coordinates... coordinates) noexcept {
   Index i = 0;

   for (auto x : {coordinates...}) {
      this->interiorEnd[i] = x;
      i++;
   }
}

__DISTRIBUTED_GRID_TEMPLATE__
void __DISTRIBUTED_GRID_PREFIX__::setInteriorEnd(const Container<Dimension, Index> &coordinates) noexcept {
   this->interiorEnd = coordinates;
}

__DISTRIBUTED_GRID_TEMPLATE__
__cuda_callable__ __DISTRIBUTED_GRID_PREFIX__::Container<Dimension, Index>
__DISTRIBUTED_GRID_PREFIX__::getInteriorEnd() const {
   return this->interiorEnd;
}
}  // namespace Meshes
}  // namespace TNL
