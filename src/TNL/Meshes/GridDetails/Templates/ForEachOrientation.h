
#pragma once

#include <TNL/Meshes/GridDetails/BasisGetter.h>

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
  using BasisGetterType = BasisGetter<Index, EntityDimension, Dimension>;

  public:
   template <typename Func>
   inline static void exec(Func func) {
      func(std::integral_constant<int, Orientation>(), BasisGetterType::template getBasis<Orientation>());

      _ForEachOrientationMain<Index, Orientation - 1, EntityDimension, Dimension, SkipValue>::exec(func);
   }
};

template <typename Index, int EntityDimension, int Dimension, int SkipValue>
struct _ForEachOrientationSupport<Index, 0, EntityDimension, Dimension, SkipValue> {
  public:
   using BasisGetterType = BasisGetter<Index, EntityDimension, Dimension>;

   template <typename Func>
   inline static void exec(Func func) {
      func(std::integral_constant<int, 0>(), BasisGetterType::template getBasis<0>());
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
