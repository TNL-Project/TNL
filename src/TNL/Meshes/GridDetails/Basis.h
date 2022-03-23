
#pragma once

#include <TNL/Meshes/GridDetails/Templates/Permutations.h>
#include <TNL/Meshes/GridDetails/Templates/Templates.h>

namespace TNL {
namespace Meshes {

template <typename Index,
          Index Orientation,
          Index EntityDimension,
          Index GridDimension,
          std::enable_if_t<Templates::isInClosedInterval(0, EntityDimension, GridDimension), bool> = true,
          std::enable_if_t<Templates::isInLeftClosedRightOpenInterval(0, Orientation, Templates::combination(EntityDimension, GridDimension)), bool> = true>
struct Basis {
   public:
      using Coordinate = TNL::Containers::StaticVector<GridDimension, Index>;
      using Value = Templates::get<
         Orientation,
         Templates::make_int_permutations<
            GridDimension,
            Templates::build_ones_pack<GridDimension - EntityDimension, GridDimension>
         >
      >;

      constexpr static Coordinate getBasis() {
         return BuildBasis<Value>::build();
      }
   private:
      template <class>
      struct BuildBasis;

      template <int... Values>
      struct BuildBasis<TNL::Meshes::Templates::int_pack<Values...>> {
        public:
         constexpr static Coordinate build() { return Coordinate(Values...); }
      };
};

}  // namespace Meshes
}  // namespace TNL
