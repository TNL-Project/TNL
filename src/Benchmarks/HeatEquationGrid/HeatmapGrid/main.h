
#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>


// Utils

// Thanks for implementation idea https://www.fluentcpp.com/2021/04/30/how-to-implement-stdconjunction-and-stddisjunction-in-c11/
// TODO: Implicit bool looks strange because, then I need to write ::value everywhere...

template<bool ...> struct bool_pack {};

template <bool... Bs>
using conjunction = std::is_same<bool_pack<true, Bs...>, bool_pack<Bs..., true>>;

// Any N dimensional grid:
//   1. Should provide local|remote step
//   2. Should know its own limits. Count of the elements at each dimension

template<int Dimension,
         typename Index,
         typename Device = TNL::Devices::Host>
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
      }

      template <typename... DimensionIndex,
                typename = std::enable_if_t<conjunction<std::is_same<Index, DimensionIndex>::value...>::value>>
      std::vector<Index> getDimensions(DimensionIndex... indicies) const noexcept {
         std::vector<Index> index{ indicies... }, result;

         std::transform(index.begin(), index.end(), std::back_inserter(result), [this](const Index& index) {
            return this -> dimensions[index];
         });

         return result;
      }

   private:
      Index dimensions[Dimension];
};

int main(int argc, char *argv[]) {
   Grid<4, int> grid;

   grid.setDimensions(1, 2, 3, 4);

   auto dimensions = grid.getDimensions(0, 1, 2);

   std::cout << dimensions[0] << dimensions[2] << dimensions[1] << std::endl;


   // const int dimensions = 4;

   // int sizes[dimensions] = { 3, 3, 3, 3 };
   // int limits[dimensions] = { 0 };


   return 0;
}
