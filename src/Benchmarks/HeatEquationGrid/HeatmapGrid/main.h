
#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <array>

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>


// Utils

template<bool ...> struct bool_pack {};

template <bool... Bs>
using conjunction = std::is_same<bool_pack<true, Bs...>, bool_pack<Bs..., true>>;

// Any N dimensional grid:
//   1. Should provide local|remote step
//   2. Should know its own limits. Count of the elements at each dimension

template<int Dimension,
         typename Index,
         typename Device = TNL::Devices::Host,
         typename = std::enable_if_t<(Dimension > 0)>>
class Grid {
   public:
      Grid() {}
      ~Grid() {}

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

      // TODO: - Add __cuda_callable__
      template <typename... DimensionIndex,
                typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>>
      std::vector<Index> getDimensions(DimensionIndex... indicies) const noexcept {
         std::vector<Index> index{ indicies... }, result;

         std::transform(index.begin(), index.end(), std::back_inserter(result), [this](const Index& index) {
            return this -> dimensions[index];
         });

         return result;
      }

      // TODO: - Implement typechecking for function
      template <typename Function,
                typename... FunctionArgs>
      void traverse(const Index dimension, Function function, FunctionArgs... args) const noexcept {

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

         for (int i = 0; i < 1 << Dimension; i++) {
            std::cout << dimensionMap[i] << " ";
         }

         std::cout << std::endl;
      }
};

int main(int argc, char *argv[]) {
   Grid<3, int> grid;

   grid.setDimensions(1, 2, 3);

  // auto dimensions = grid.getDimensions(0, 1, 2);

   return 0;
}
