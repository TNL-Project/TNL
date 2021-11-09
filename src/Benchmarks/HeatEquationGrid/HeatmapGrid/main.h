
#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <array>

#include "../Base/HeatmapSolver.h"

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/ParallelFor.h>

#pragma once

template<bool ...> struct bool_pack {};

template <bool... Bs>
using conjunction = std::is_same<bool_pack<true, Bs...>, bool_pack<Bs..., true>>;

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
       *  @param[in] dimensions - A parameter pack, which specifies edges count in the specific dimension
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
                typename = std::enable_if_t<sizeof...(DimensionIndex) > 0>>
      std::vector<Index> getDimensions(DimensionIndex... indicies) const noexcept {
         std::vector<Index> dimensionIndicies{ indicies... }, result;

         std::transform(dimensionIndicies.begin(),
                        dimensionIndicies.end(),
                        std::back_inserter(result), [this](const Index& index) {
            return this -> getDimension(index);
         });

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
                typename = std::enable_if_t<sizeof...(DimensionIndex) > 0>>
      std::vector<Index> getEntitiesCounts(DimensionIndex... indicies) const noexcept
      {
         std::vector<Index> dimensionIndicies{indicies...}, result;

         std::transform(dimensionIndicies.begin(),
                        dimensionIndicies.end(),
                        std::back_inserter(result), [this](const Index &index) {
            return getEntitiesCount(index);
         });

         return result;
      }
      /**
       * @brief - Traversers a specified dimension in parallel.
       */
      template <typename Function, typename... FunctionArgs>
      void traverse(const Index dimension, Function function, FunctionArgs... args) const noexcept {
         auto entitiesCount = getEntitiesCount(dimension)[0];
         auto identity = [] __cuda_callable__ (const Index&& index) mutable { return index };

         traverse(0, entitiesCount, identity, function, args...);
      }
      /**
       * TODO: - A possibility of improvement, as it should be possible to specify more precise functions
       *         to remove user knowledge of functionality
       */
      template <typename Function, typename IndexTransform, typename... FunctionArgs>
      void traverse(const Index start,
                    const Index end,
                    IndexTransform transform,
                    Function function,
                    FunctionArgs... args) const noexcept {
         TNL_ASSERT_LT(startIndex, endIndex, "Start index must be less than endIndex")

         auto lambda = [=] __cuda_callable__(const Index index, FunctionArgs... args) mutable
         {
            auto transformedIndex = transform(index);

            function(transformedIndex, args...);
         };

         TNL::Algorithms::ParallelFor<Device>::exec(startIndex, endIndex, lambda, args...);
      }
   private:
      std::array<Index, Dimension> dimensions;
      /**
       * @brief - A dimension map is a store for dimension limits over all combinations of basis.
       *          First, (n choose 0) elements will contain the count of 0 dimension elements
       *          Second, (n choose 1) elements will contain the count of 1-dimension elements
       *          ....
       *
       *          For example, let's have a 3-d grid, then the map indexing will be the next:
       *            0 - 0 - count of vertices
       *            1, 2, 3 - count of edges in x, y, z plane
       *            4, 5, 6 - count of faces in xy, xz, yz plane
       *            7 - count of cells in xyz plane
       *
       * @warning - The ordering of is lexigraphical.
       */
      std::array<Index, 1 << Dimension> dimensionMap;
      /**
       * @brief - A cumulative map over dimensions.
       */
      std::array<Index, Dimension + 1> cumulativeDimensionMap;
      /**
       * @brief - Fills dimensions map for N-dimensional Grid.
       *
       * @complexity - O(2 ^ Dimension)
       */
      void refreshDimensionMaps() noexcept {
         std::array<bool, Dimension> combinationBuffer = {};
         std::size_t j = 0;

         std::fill(cumulativeDimensionMap.begin(), cumulativeDimensionMap.end(), 0);

         for (std::size_t i = 0; i <= Dimension; i++) {
            std::fill(combinationBuffer.begin(), combinationBuffer.end(), false);
            std::fill(combinationBuffer.end() - i, combinationBuffer.end(), true);

            do {
               int result = 1;

               for (std::size_t i = 0; i < combinationBuffer.size(); i++)
                  result *= combinationBuffer[i] ? dimensions[i] : dimensions[i] + 1;

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

   auto uxView = ux.getView(), auxView = aux.getView();

   TNL::Containers::Array<Real, Device> ux(entitiesCount), // data at step u
                                        aux(entitiesCount);// data at step u + 1

   // Invalidate ux/aux
   ux = 0;
   aux = 0;



   return false;
};

int main(int argc, char *argv[]) {
   Grid<3, int> grid;

   grid.setDimensions(1, 2, 3);



   return 0;
}
