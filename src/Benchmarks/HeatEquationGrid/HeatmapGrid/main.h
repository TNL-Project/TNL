
#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <array>
#include <bitset>

#include "../Base/HeatmapSolver.h"

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/ParallelFor.h>

#pragma once

template<bool ...> struct bool_pack {};

template <bool... Bs>
using conjunction = std::is_same<bool_pack<true, Bs...>, bool_pack<Bs..., true>>;

template<int Dimension,
         typename Index,
         typename = std::enable_if_t<(Dimension > 0)>,
         typename = std::enable_if_t<std::is_integral<Index>::value>>
using Container = TNL::Containers::StaticArray<Dimension, Index>;

template<int Dimension,
         typename Index,
         typename = std::enable_if_t<(Dimension > 0)>,
         typename = std::enable_if_t<std::is_integral<Index>::value>>
class GridEntity {
   public:
      __cuda_callable__ inline explicit
      GridEntity(const Container<Dimension, Index>& startPosition,
                 const Index& startOffset,
                 const Index& offset,
                 const Container<Dimension, Index>& dimensions,
                 const Container<Dimension, bool> direction) :
                 startPosition(startPosition),
                 startOffset(startOffset),
                 offset(offset),
                 dimensions(dimensions),
                 direction(direction) {
        this -> position = -1;
      };

      __cuda_callable__
      ~GridEntity() {};

      __cuda_callable__ inline
      Container<Dimension, Index> getPosition() noexcept {
         if (position[0] == -1) {
            auto position = startPosition;
            Index tmpOffset = offset;

            Index dim = Dimension - 1;

            while (tmpOffset) {
               Index newIndex = position[dim] + tmpOffset, dimension = dimensions[dim];
               Index quotient = newIndex / dimension;
               Index reminder = newIndex - (dimension * quotient);

               position[dim] = reminder;
               tmpOffset = quotient;

               dim -= 1;
            }

            this -> position = position;
         }

         return position;
      }

      __cuda_callable__ inline
      Index getIndex() const noexcept {
         return startOffset + offset;
      }
   private:
      const Container<Dimension, Index> startPosition;
      const Index startOffset;

      Container<Dimension, Index> position;
      const Index offset;

      const Container<Dimension, Index> dimensions;
      const Container<Dimension, bool> direction;
};

template<int Dimension,
         typename Index,
         typename Device = TNL::Devices::Host,
         typename = std::enable_if_t<(Dimension > 0)>>
class Grid {
   public:
      Grid() {}
      ~Grid() {}

      /**
       *  @brief - Specifies dimensions of the grid
       *  @param[in] dimensions - A parameter pack, which specifies points count in the specific dimension.
       *                          Most significant dimension is in the beginning of the list.
       *                          Least significant dimension is in the end of the list
       */
      template <typename... Dimensions,
                typename = std::enable_if_t<conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
      void setDimensions(Dimensions... dimensions) noexcept {
         Index i = 0;

         for (auto x: { dimensions... }) {
            TNL_ASSERT_GT(x, 0, "Dimension must be positive");
            this -> dimensions[i] = x;
            i++;
         }

         refreshDimensionMaps();
      }
      /**
       * @param[in] index - index of dimension
       */
      inline __cuda_callable__ Index getDimension(Index index) const noexcept {
         TNL_ASSERT_GT(index, 0, "Index must be greater than zero");
         TNL_ASSERT_LT(index, Dimension, "Index must be less than Dimension");

         return dimensions[index];
      }
      /**
       * @param[in] indicies - A dimension index pack
       */
      template <typename... DimensionIndex,
                typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
                typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
      Container<sizeof...(DimensionIndex), Index> getDimensions(DimensionIndex... indicies) const noexcept {
         TNL::Containers::StaticArray<sizeof...(DimensionIndex), Index> result{indicies...};

         for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
            result[i] = this -> getDimension(result[i]);

         return result;
      }
      /**
       * @param[in] index - index of dimension
       */
      inline __cuda_callable__ Index getEntitiesCount(Index index) const noexcept {
         TNL_ASSERT_GT(index, 0, "Index must be greater than zero");
         TNL_ASSERT_LE(index, Dimension, "Index must be less than or equal to Dimension");

         return cumulativeDimensionMap(index);
      }
      /**
       * @brief - Returns the number of entities of specific dimension
       */
      template <typename... DimensionIndex,
                typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
                typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
      Container<sizeof...(DimensionIndex), Index> getEntitiesCounts(DimensionIndex... indicies) const noexcept {
         Container<sizeof...(DimensionIndex), Index> result{indicies...};

         for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
            result[i] = this -> getEntitiesCount(result[i]);

         return result;
      }
      /**
       * @brief - Traversers all elements in the grid
       */
      template <typename Function, typename... FunctionArgs>
      void traverseAll(Function function, FunctionArgs... args) const noexcept {
         auto lambda = [=] __cuda_callable__(const Index index, FunctionArgs... args) mutable {
            function(index, args...);
         };

         TNL::Algorithms::ParallelFor<Device>::exec(0, cumulativeDimensionMap[0], lambda, args...);
      }
      /**
       * @brief - Traverses a grid from start index to end index.
       *
       * @param[in] start - a start index of point
       * @param[in] end - an end index of point
       * @param[in] directions - A pack of boolean vector flags with the size of the dimension.
       *   For example, let's have the 3-dimensional grid.
       *     A pack {false, false, false} will call function for all points
       *     A pack {true, false, false} will call function for edges directed over dimension at index of true
       *     A pack {true, true, false} will call function for faces directed over dimension at index of true
       *     A pack {true, true, true} will call function for cells directed over
       *
       */
      template<typename Function, typename... FunctionArgs>
      void traverse(const Container<Dimension, Index>& start,
                    const Container<Dimension, Index>& end,
                    const TNL::Containers::Array<Container<Dimension, bool>, Device>& directions,
                    Function function,
                    FunctionArgs... args) const noexcept {
         // TODO: - This will overflow for higher dimensions
         Index startCollapsedIndex = 0;
         Index endCollapsedIndex = 0;

         // TODO: - For higher dimensions, this will overflow.
         Container<Dimension, Index> multipliers = 1;

         for (Index i = 0; i < Dimension; i++) {
            for (Index j = i + 1; j < Dimension; j++)
               multipliers[i] *= dimensions[j];

            startCollapsedIndex += start[i] * multipliers[i];
            endCollapsedIndex += end[i] * multipliers[i];
         }

         // TODO: - Improve message formatting
         TNL_ASSERT_LT(startCollapsedIndex, endCollapsedIndex, "Traverse range must be in [start..<end]");
         TNL_ASSERT_LE(endCollapsedIndex, this -> cumulativeDimensionMap[0], "End must be less, than amount of points in grid");

         auto dimensions = this -> dimensions;
         auto directionsView = directions.getConstView();

         auto outerFunction = [=] __cuda_callable__ (Index offset, FunctionArgs... args) mutable {
            for (Index j = 0; j < directionsView.getSize(); j++) {
               GridEntity<Dimension, Index> entity{ start, startCollapsedIndex, offset, dimensions, directionsView[j] };

               function(entity, args...);
            }
         };

         TNL::Algorithms::ParallelFor<Device>::exec(0, endCollapsedIndex - startCollapsedIndex + 1, outerFunction, args...);
      }
   private:
      Container<Dimension, Index> dimensions;
      /**
       * @brief - A dimension map is a store for dimension limits over all combinations of basis.
       *          First, (n choose 0) elements will contain the count of 0 dimension elements
       *          Second, (n choose 1) elements will contain the count of 1-dimension elements
       *          ....
       *
       *          For example, let's have a 3-d grid, then the map indexing will be the next:
       *            0 - 0 - count of vertices
       *            1, 2, 3 - count of edges in z, y, x plane
       *            4, 5, 6 - count of faces in yz, xz, xy plane
       *            7 - count of cells in z y x plane
       *
       * @warning - The ordering of is lexigraphical.
       */
      Container<1 << Dimension, Index> dimensionMap;
      /**
       * @brief - A cumulative map over dimensions.
       */
      Container<Dimension + 1, Index> cumulativeDimensionMap;
      /**
       * @brief - Fills dimensions map for N-dimensional Grid.
       *
       * @complexity - O(2 ^ Dimension)
       */
      void refreshDimensionMaps() noexcept {
         std::array<bool, Dimension> combinationBuffer = {};
         std::size_t j = 0;

         for (std::size_t i = 0; i < Dimension + 1; i++)
            cumulativeDimensionMap[i] = 0;

         for (std::size_t i = 0; i <= Dimension; i++) {
            std::fill(combinationBuffer.begin(), combinationBuffer.end(), false);
            std::fill(combinationBuffer.end() - i, combinationBuffer.end(), true);

            do {
               int result = 1;

               for (std::size_t k = 0; k < combinationBuffer.size(); k++)
                  result *= combinationBuffer[k] ? dimensions[k] - 1 : dimensions[k];

               dimensionMap[j] = result;
               cumulativeDimensionMap[i] += result;

               j++;
            } while (std::next_permutation(combinationBuffer.begin(), combinationBuffer.end()));
         }
      }
};

template <typename Real>
template <typename Device>
bool HeatmapSolver<Real>::solve(const HeatmapSolver<Real>::Parameters &params) const {
   Grid<2, int, Device> grid;

   grid.setDimensions(params.xSize, params.ySize);

   const Real hx = params.xDomainSize / (Real)grid.getDimension(0);
   const Real hy = params.yDomainSize / (Real)grid.getDimension(1);
   const Real hx_inv = 1 / (hx * hx);
   const Real hy_inv = 1 / (hy * hy);

   auto entitiesCount = grid.getEntitiesCount(0);
   auto timestep = params.timeStep ? params.timeStep : std::min(hx * hx, hy * hy);

   TNL::Containers::Array<Real, Device> ux(entitiesCount), // data at step u
                                        aux(entitiesCount);// data at step u + 1

   auto uxView = ux.getView(), auxView = aux.getView();

   // Invalidate ux/aux
   ux = 0;
   aux = 0;

  /* auto init = [=] __cuda_callable__(const auto& entity) mutable {
      // auto index = j * params.xSize + i;

      //auto x = i * hx - params.xDomainSize / 2;
      //auto y = j * hy - params.yDomainSize / 2;

      //uxView[index] = exp(params.sigma * (x * x + y * y));
   };*/

  /* grid.traverse({ 1, 1 },
                 { grid.getDimension(0) - 1, grid.getDimension(1) - 1 },
                 { { 0b00 } },
                 init);*/

   return false;
};

int main(int argc, char *argv[]) {
   Grid<3, int, TNL::Devices::Host> grid;

   grid.setDimensions(3, 3, 3);

   auto fn = [=] __cuda_callable__ (int index) {
      //printf("%d\n", index);
   };

   grid.traverseAll(fn);

   auto fn_entity = [=] __cuda_callable__ (GridEntity<3, int> entity) {
      printf("%d \n", entity.getIndex());
   };

   Container<3, int> direction { 0, 0, 0};

   grid.traverse({ 1, 1, 1 }, { 2, 2, 2 }, { direction }, fn_entity);

   return 0;
}
