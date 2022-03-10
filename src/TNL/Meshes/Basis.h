
#pragma once

#include <type_traits>
#include <TNL/Meshes/Templates.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Meshes {

namespace Templates {
template <class>
struct BuildBasis;

template <int... Values>
struct BuildBasis<TNL::Meshes::Templates::int_pack<Values...>> {
  public:
   constexpr static Coordinate build() { return Coordinate(Values...); }
};
}  // namespace Templates

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
            Templates::build_ones_pack<EntityDimension, GridDimension>
         >
      >;

      constexpr static Coordinate getBasis() {
         return Templates::BuildBasis<Value>::build();
      }

};

}  // namespace Meshes
}  // namespace TNL
