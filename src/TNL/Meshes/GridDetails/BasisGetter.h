
#pragma once

#include <TNL/Meshes/GridDetails/Templates/Permutations.h>
#include <TNL/Meshes/GridDetails/Templates/DescendingFor.h>
#include <TNL/Meshes/GridDetails/Templates/Templates.h>

namespace TNL {
namespace Meshes {

template <typename Index,
          Index EntityDimension,
          Index GridDimension>
struct BasisGetter {
   public:
      using Coordinate = TNL::Containers::StaticVector<GridDimension, Index>;
      using OrientationBasesContainer = TNL::Containers::StaticVector<Templates::combination(EntityDimension, GridDimension), Coordinate>;
      using Permutations = Templates::make_int_permutations<
         GridDimension,
         Templates::build_ones_pack<GridDimension - EntityDimension, GridDimension>
      >;

      template<int Orientation,
               std::enable_if_t<Templates::isInLeftClosedRightOpenInterval(0, Orientation, Templates::combination(EntityDimension, GridDimension)), bool> = true>
      constexpr static Coordinate getBasis() {
         using Value = Templates::get<Orientation, Permutations>;

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
