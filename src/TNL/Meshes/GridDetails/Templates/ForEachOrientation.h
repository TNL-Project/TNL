
#pragma once

#include <TNL/Meshes/GridDetails/Basis.h>

namespace TNL {
namespace Meshes {
namespace Templates {

template <typename Index,
          int Orientation,
          int EntityDimension,
          int Dimension,
          int SkipValue>
struct _ForEachOrientationMain;

template <typename Index, int Orientation, int EntityDimension, int Dimension, int SkipValue>
struct _ForEachOrientationSupport {
  public:
   template <typename Func>
   inline static void exec(Func func) {
      using Basis = Basis<Index, Orientation, EntityDimension, Dimension>;

      func(std::integral_constant<int, Orientation>(), Basis::getBasis());

      _ForEachOrientationMain<Index, Orientation - 1, EntityDimension, Dimension, SkipValue>::exec(func);
   }
};

template <typename Index, int EntityDimension, int Dimension, int SkipValue>
struct _ForEachOrientationSupport<Index, 0, EntityDimension, Dimension, SkipValue> {
  public:
   template <typename Func>
   inline static void exec(Func func) {
      using Basis = Basis<Index, 0, EntityDimension, Dimension>;

      func(std::integral_constant<int, 0>(), Basis::getBasis());
   }
};

template <typename Index, int EntityDimension, int Dimension>
struct _ForEachOrientationSupport<Index, 0, EntityDimension, Dimension, 0> {
  public:
   template <typename Func>
   inline static void exec(Func func) {}
};

template <typename Index, int Orientation, int EntityDimension, int Dimension, int SkipValue>
struct _ForEachOrientationMain
    : std::conditional_t<Orientation == SkipValue,
                         _ForEachOrientationSupport<Index, (Orientation <= 1 ? 0 : Orientation - 1), EntityDimension, Dimension, SkipValue>,
                         _ForEachOrientationSupport<Index, Orientation, EntityDimension, Dimension, SkipValue>> {};

template <typename Index, int EntityDimension, int Dimension, int skipOrientation = -1>
struct ForEachOrientation : _ForEachOrientationMain<Index,
                                                    combination(EntityDimension, Dimension) - 1,
                                                    EntityDimension,
                                                    Dimension,
                                                    skipOrientation> {};
}  // namespace Templates
}  // namespace Meshes
}  // namespace TNL
