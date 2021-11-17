
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
      struct Context {
         public:
            const int startOffset;

            const Container<Dimension, Index> startPosition;
            const Container<Dimension, Index> traverseRectDimensions;
            const Container<Dimension, bool> direction;

            Context(const int startOffset,
                    const Container<Dimension, Index> startPosition,
                    const Container<Dimension, Index> traverseRectDimensions,
                    const Container<Dimension, bool> direction):
                    startOffset(startOffset),
                    startPosition(startPosition),
                    traverseRectDimensions(traverseRectDimensions),
                    direction(direction) {}
      };

      __cuda_callable__
      inline explicit GridEntity(const Context& context,
                                 const Index& vertexIndex) :
                                 vertexIndex(vertexIndex),
                                 context(context) {
        this -> position = -1;
      };

      __cuda_callable__
      ~GridEntity() {};

      __cuda_callable__
      inline Container<Dimension, Index> getPosition() noexcept {
         if (position[0] == -1) {
            Container<Dimension, Index> position = 0;
            Index tmpOffset = vertexIndex - context.startOffset;

            Index dim = Dimension - 1;

            while (tmpOffset) {
               Index newIndex = position[dim] + tmpOffset, dimension = context.traverseRectDimensions[dim];
               Index quotient = newIndex / dimension;
               Index reminder = newIndex - (dimension * quotient);

               position[dim] = reminder;
               tmpOffset = quotient;

               dim -= 1;
            }

            for (Index i = 0; i < Dimension; i++)
               this -> position[i] = position[i] + context.startPosition[i];
         }

         return position;
      }

      __cuda_callable__
      inline Index getIndex() const noexcept {
         return vertexIndex;
      }
   private:
      Container<Dimension, Index> position;

      const Index vertexIndex;
      const Context& context;
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
         TNL_ASSERT_GE(index, 0, "Index must be greater than zero");
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
         Container<sizeof...(DimensionIndex), Index> result{indicies...};

         for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
            result[i] = this -> getDimension(result[i]);

         return result;
      }
      /**
       * @param[in] index - index of dimension
       */
      inline __cuda_callable__ Index getEntitiesCount(Index index) const noexcept {
         TNL_ASSERT_GE(index, 0, "Index must be greater than zero");
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
       * @param[in] index - index of dimension
       */
      inline __cuda_callable__ Index getEndIndex(Index index) const noexcept {
         TNL_ASSERT_GE(index, 0, "Index must be greater than zero");
         TNL_ASSERT_LT(index, Dimension, "Index must be less than or equal to Dimension");

         return this -> getDimension(index) - 1;
      }
       /**
       * @brief - Returns the last index of specific dimensions
       */
      template <typename... DimensionIndex,
         typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>,
         typename = std::enable_if_t<(sizeof...(DimensionIndex) > 0)>>
         Container<sizeof...(DimensionIndex), Index> getEndIndicies(DimensionIndex... indicies) const noexcept {
         Container<sizeof...(DimensionIndex), Index> result{ indicies... };

         for (std::size_t i = 0; i < sizeof...(DimensionIndex); i++)
            result[i] = this->getEndIndex(result[i]);

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
      void traverse(const Container<Dimension, Index>& firstVertex,
                    const Container<Dimension, Index>& secondVertex,
                    const TNL::Containers::Array<Container<Dimension, bool>, Device>& directions,
                    Function function,
                    FunctionArgs... args) const noexcept {
         Index verticesCount = 1, offset = 0;
         Container<Dimension, Index> traverseRectDimensions, offsetPosition, multipliers;

         for (Index i = 0; i < Dimension; i++) {
            traverseRectDimensions[i] = abs(secondVertex[i] - firstVertex[i]) + 1;
            verticesCount *= traverseRectDimensions[i];
            offsetPosition[i] = std::min(firstVertex[i], secondVertex[i]);

            multipliers[i] = i == 0 ? 1 : multipliers[i - 1] * dimensions[i];
            offset += offsetPosition[i] * multipliers[i];

            TNL_ASSERT_LT(firstVertex[i], dimensions[i], "End index must be in dimensions range");
            TNL_ASSERT_LT(secondVertex[i], dimensions[i], "Start index must be in dimensions range");
         }

         using Context = typename GridEntity<Dimension, Index>::Context;

         auto outerFunction = [function] __cuda_callable__(Index offset, const Context& context, FunctionArgs... args) mutable {
            GridEntity<Dimension, Index> entity{ context, context.startOffset + offset };

            function(entity, args...);
         };

         Index lowerBound = 0, upperBound = verticesCount;

         for (Index i = 0; i < directions.getSize(); i++) {
            const Context context{ offset, offsetPosition, traverseRectDimensions, directions[i] };

            TNL::Algorithms::ParallelFor<Device>::exec(lowerBound, upperBound, outerFunction, context, args...);
         }
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

   auto init = [=] __cuda_callable__(GridEntity<2, int> entity) mutable {
      auto position = entity.getPosition();
      auto index = entity.getIndex();

      auto x = position[0] * hx - params.xDomainSize / 2;
      auto y = position[1] * hx - params.yDomainSize / 2;

      uxView[index] = exp(params.sigma * (x * x + y * y));
   };

   const Container<2, bool> direction{ false, false };

   grid.traverse({ 1, 1 },
                 { grid.getEndIndex(0) - 1, grid.getEndIndex(1) - 1 },
                 { direction },
                 init);

   if (!writeGNUPlot("data.txt", params, ux))
      return false;

   auto xDimension = grid.getDimension(0);

   auto next = [=] __cuda_callable__(const GridEntity<2, int>& entity) mutable {
      auto index = entity.getIndex();

      auxView[index] = (uxView[index - 1] - 2 * uxView[index] + uxView[index + 1]) * hx_inv +
                       (uxView[index - xDimension] - 2 * uxView[index] + uxView[index + xDimension]) * hy_inv;
   };

   auto update = [=] __cuda_callable__(int i) mutable {
      uxView[i] += auxView[i] * timestep;
   };

   Real start = 0;

   while (start < params.finalTime) {
      grid.traverse({ 1, 1 },
                    { grid.getEndIndex(0) - 1, grid.getEndIndex(1) - 1 },
                    { direction },
                    next);

      grid.traverseAll(update);

      start += timestep;
   }

   return writeGNUPlot("data_final.txt", params, ux);
};

int main(int argc, char* argv[]) {
   using Real = double;

   auto config = HeatmapSolver<Real>::Parameters::makeInputConfig();

   TNL::Config::ParameterContainer parameters;
   if (!parseCommandLine(argc, argv, config, parameters))
      return EXIT_FAILURE;

   auto device = parameters.getParameter<TNL::String>("device");
   auto params = HeatmapSolver<Real>::Parameters(parameters);

   HeatmapSolver<Real> solver;

   if (device == "host" && !solver.solve<TNL::Devices::Host>(params))
      return EXIT_FAILURE;

#ifdef HAVE_CUDA
   if (device == "cuda" && !solver.solve<TNL::Devices::Cuda>(params))
      return EXIT_FAILURE;
#endif

   return EXIT_SUCCESS;
}

/*
int main(int argc, char *argv[]) {
   Grid<3, int, TNL::Devices::Host> grid;

   grid.setDimensions(3, 3, 3);

   auto fn = [=] __cuda_callable__ (int index) {
      //printf("%d\n", index);
   };

   grid.traverseAll(fn);

   auto fn_entity = [=] __cuda_callable__ (GridEntity<3, int> entity) {
      printf("%d %d %d \n", entity.getPosition()[0], entity.getPosition()[1], entity.getPosition()[2]);
   };

   Container<3, int> direction { 0, 0, 0 };

   grid.traverse({ 1, 1, 1 }, { 1, 1, 1 }, { direction }, fn_entity);

   return 0;
}
*/
