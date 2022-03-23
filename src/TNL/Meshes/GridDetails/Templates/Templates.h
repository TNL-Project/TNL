
#pragma once

#include <type_traits>

namespace TNL {
namespace Meshes {
namespace Templates {

template <size_t Value, size_t Power>
constexpr size_t pow() {
   size_t result = 1;

   for (size_t i = 0; i < Power; i++) result *= Value;

   return result;
}

template <typename Index>
constexpr Index product(Index from, Index to) {
   Index result = 1;

   if (from <= to)
      for (Index i = from; i <= to; i++) result *= i;

   return result;
}

template <typename Index>
constexpr Index combination(Index k, Index n) {
   return product(k + 1, n) / product(1, n - k);
}

template <typename Index>
constexpr Index firstKCombinationSum(Index k, Index n) {
   if (k == 0) return 0;

   if (k == n) return 1 << n;

   Index result = 0;

   // Fraction simplification of k-combination
   for (Index i = 0; i < k; i++) result += combination(i, n);

   return result;
}

constexpr bool isInClosedInterval(int lower, int value, int upper) { return lower <= value && value <= upper; }

constexpr bool isInLeftClosedRightOpenInterval(int lower, int value, int upper) { return lower <= value && value < upper; }

}  // namespace Templates
}  // namespace Meshes
}  // namespace TNL
