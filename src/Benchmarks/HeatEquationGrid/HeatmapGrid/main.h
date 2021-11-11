
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
#include <TNL/Containers/Vector.h>
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
      GridEntity(const Container<Dimension, Index>& origin,
                 const Container<Dimension, Index>& dimensions,
                 const std::bitset<Dimension>& direction): origin(origin),
                                                           dimensions(dimensions),
                                                           direction(direction) {};
   private:
      Container<Dimension, Index> origin;
      Container<Dimension, Index> dimensions;
      std::bitset<Dimension> direction;
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
       *  @param[in] dimensions - A parameter pack, which specifies edges count in the specific dimension.
       *                          Most significant dimension is in the beginning of the list.
       *                          Least significant dimension is in the end of the list
       */
      template <typename... Dimensions,
                typename = std::enable_if_t<conjunction<std::is_same<Index, Dimensions>::value...>::value>,
                typename = std::enable_if_t<sizeof...(Dimensions) == Dimension>>
      void setDimensions(Dimensions... dimensions) noexcept {
         Index i = 0;

         for (auto x: { dimensions... }) {
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
       *
       * - TODO: - Add __cuda_callable__
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
                    const TNL::Containers::Vector<std::bitset<Dimension>>& directions,
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
         TNL_ASSERT_LT(startCollapsedIndex, endCollapsedIndex, "Traverse range must be in [start..<end]")
         TNL_ASSERT_LE(endCollapsedIndex, this -> cumulativeDimensionMap[0], "End must be less, than amount of points in grid");

         auto dimensions = this -> dimensions;

         auto outerFunction = [=] __cuda_callable__ (const Index i, FunctionArgs... args) mutable {
            Index dim = Dimension - 1;

            while (i != 0) {
               Index newIndex = start[dim] + i, dimension = dimensions[dim];
               Index quotient = newIndex / dimension;
               Index reminder = newIndex - (dimension * quotient);

               start[dim] = reminder;
               i = quotient;

               dim -= 1;
            }

            for (const auto& direction: directions) {
               GridEntity<Dimension, Index> entity = { start, dimensions, direction };

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
                  result *= combinationBuffer[k] ? dimensions[k] : dimensions[k] + 1;

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



   return false;
};

int main(int argc, char *argv[]) {
   Grid<3, int> grid;

   grid.setDimensions(3, 2, 1);

   return 0;
}
