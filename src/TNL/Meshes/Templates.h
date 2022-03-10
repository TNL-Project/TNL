
#pragma once

#include <TNL/Algorithms/ParallelFor.h>

#include <type_traits>

namespace TNL {
namespace Meshes {
namespace Templates {

/**
 * One of the possible implementation of the conjuction operator.
 *
 * This one is taken from https://en.cppreference.com/w/cpp/types/conjunction
 */

template <class...>
struct conjuction : std::true_type {};

template <class Type>
struct conjuction<Type> : Type {};

template <class Head, class... Tail>
struct conjuction<Head, Tail...> : std::conditional_t<bool(Head::value), conjuction<Tail...>, Head> {};

template <class... Types>
constexpr bool conjunction_v = conjuction<Types...>::value;

/**
 * One of the possible implementation of the conjuction operator.
 *
 * This one is taken from https://en.cppreference.com/w/cpp/types/disjunction
 */

template <class...>
struct disjunction : std::false_type {};

template <class Type>
struct disjunction<Type> : Type {};

template <class Head, class... Tail>
struct disjunction<Head, Tail...> : std::conditional_t<bool(Head::value), Head, disjunction<Tail...>> {};

template <class... Types>
constexpr bool disjunction_v = disjunction<Types...>::value;

/*
 * A pack of type and its count
 */
template <class, size_t>
struct counted_pack {};

/*
 * A pack of types
 */
template <class...>
struct pack {};

/*
 * A pack of int values
 */
template <int...>
struct int_pack {};

/**
 * A merge operation between to packs.
 *
 * Merge<type> -> type
 * Merge<pack<A>, pack<B>> -> pack<A, B>
 * Merge<int_pack<A>, int_pack<B>> -> int_pack<A, B>
 * Merge<pack<A_1>, pack<A_2>, ..., pack<A_n>> -> pack<A_1, ..., A_n>
 * Merge<int_pack<A_1>, int_pack<A_2>, ..., int_pack<A_n>> -> int_pack<A_1, ..., A_n>
 */

template <class... Packs>
struct Merge;

template <class P>
struct Merge<P> {
  public:
   using type = P;
};

template <class... LHS, class... RHS>
struct Merge<pack<LHS...>, pack<RHS...>> {
  public:
   using type = pack<LHS..., RHS...>;
};

template <int... LHS, int... RHS>
struct Merge<int_pack<LHS...>, int_pack<RHS...>> {
  public:
   using type = int_pack<LHS..., RHS...>;
};

template <class LHS, class... RHS>
struct Merge<LHS, RHS...> : Merge<LHS, typename Merge<RHS...>::type> {};

template <class... types>
using merge = typename Merge<types...>::type;

/**
 * Prepends a type to a parameter pack
 */

template <class, class>
struct Prepend {};

template <class Type, class... Types>
struct Prepend<Type, pack<Types...>> {
   using type = pack<Type, Types...>;
};

template <typename Type, class Pack>
using prepend = typename Prepend<Type, Pack>::type;

/**
 * Pushes type to the CountedTypes pack.
 *
 * If type exists in the CountedTypes pack, then will increment a count of type.
 * Otherwise appends a type to the CountedTypes pack.
 */

template <class, class>
struct Push;

template <class Type, class... Types, size_t... Counts>
struct Push<Type, pack<counted_pack<Types, Counts>...>> {
  public:
   using type = std::conditional_t<disjunction<std::is_same<Types, Type>...>::value,
                                   pack<counted_pack<Types, Counts + (std::is_same<Types, Type>::value ? 1 : 0)>...>,
                                   pack<counted_pack<Types, Counts>..., counted_pack<Type, 1>>>;
};

template <class Type, class CountedTypes>
using push = typename Push<Type, CountedTypes>::type;

/**
 * Removes first occurance of type in the pack
 */

template <class, class>
struct RemoveFirst;

template <class Type>
struct RemoveFirst<Type, pack<>> {
  public:
   using type = pack<>;
};

template <class Type, class CountedType>
using remove_first = typename RemoveFirst<Type, CountedType>::type;

template <class Type, class Head, class... Tail>
struct RemoveFirst<Type, pack<Head, Tail...>> {
  public:
   using type = std::conditional_t<std::is_same<Type, Head>::value, pack<Tail...>, prepend<Head, remove_first<Type, pack<Tail...>>>>;
};

/*
 * Pops type out of the CountedTypes pack.
 *
 * If type exists in the CountedTypes pack, then decrease it count by one.
 * If the count of type in the CountedTypes pack is zero, then pop it.
 */
template <class, class>
struct Pop;

template <class T, class... Types, std::size_t... Indices>
struct Pop<T, pack<counted_pack<Types, Indices>...>> {
   using type = remove_first<counted_pack<T, 0>, pack<counted_pack<Types, Indices - (std::is_same<Types, T>::value ? 1 : 0)>...>>;
};

template <class Type, class CountedType>
using pop = typename Pop<Type, CountedType>::type;

/*
 * Builds CountedTypes pack by counting the occurances of the types in input.
 */
template <class Pack, class CountedTypes = pack<>>
struct CountTypes {
   using type = CountedTypes;
};

template <class CountedTypes, class Head, class... Tail>
struct CountTypes<pack<Head, Tail...>, CountedTypes> : CountTypes<pack<Tail...>, push<Head, CountedTypes>> {};

// For int_pack count every int value as a separate type (push<int_pack<Head>)
template <class CountedTypes, int Head, int... Tail>
struct CountTypes<int_pack<Head, Tail...>, CountedTypes> : CountTypes<int_pack<Tail...>, push<int_pack<Head>, CountedTypes>> {};

template <class Pack>
using count_types = typename CountTypes<Pack>::type;

/*
 * An identity type
 */
template <class T>
struct identity {
   using type = T;
};

/**
 * Generates all permutations with repetitions of length N from the pack of types.
 *
 * Thanks for idea: https://stackoverflow.com/questions/36465889/permutation-pn-r-of-types-in-compile-time
 */

template <std::size_t, class, class = pack<>>
struct MakePermutations;

// Workaround for GCC's partial ordering failure
template <std::size_t, class, class>
struct MakePermutationsImpl;

template <std::size_t N, class... Types, std::size_t... Counts, class... Current>
struct MakePermutationsImpl<N, pack<counted_pack<Types, Counts>...>, pack<Current...>> {
   // The next item can be anything in Types...
   // We append it to Current... and pop it from the list of types, then
   // recursively generate the remaining items
   // Do this for every type in Types..., and concatenate the result.
   using type = merge<typename MakePermutations<N - 1, pop<Types, pack<counted_pack<Types, Counts>...>>, pack<Current..., Types>>::type...>;
};

template <std::size_t N, class... Types, std::size_t... Counts, class... Current>
struct MakePermutations<N, pack<counted_pack<Types, Counts>...>, pack<Current...>> {
   // Note that we don't attempt to evaluate MakePermutationsImpl<...>::type
   // until we are sure that N > 0
   using type = typename std::conditional_t<N == 0, identity<pack<pack<Current...>>>,
                                            MakePermutationsImpl<N, pack<counted_pack<Types, Counts>...>, pack<Current...>>>::type;
};

template <std::size_t N, typename Pack>
using make_permutations = typename MakePermutations<N, count_types<Pack>>::type;

/*
 * Recursively goes through pack tree and merges specified level.
 */

template <std::size_t, class...>
struct GroupLevel {};

template <std::size_t N, class... Types>
struct GroupLevel<N, pack<Types...>> {
   using type = pack<typename GroupLevel<N - 1, Types>::type...>;
};

template <class... Types>
struct GroupLevel<0, pack<Types...>> {
   using type = merge<Types...>;
};

template <std::size_t N, class... Types>
using group_level = typename GroupLevel<N, Types...>::type;

/*
 * A support for the int_pack permutations.
 * The result of make_permutations is pack<pack<int_pack<>, int_pack<>...>, ...>, that's why we merge the 1 level.
 */
template <std::size_t N, typename Pack>
using make_int_permutations = group_level<1, make_permutations<N, Pack>>;

/**
 * Builds the pack with k ones at the end.
 */
template <int, int, class = int_pack<>>
struct BuildOnesPack;

template <int OnesCount, int Size, int... Values>
struct BuildOnesPack<OnesCount, Size, int_pack<Values...>> : std::conditional_t<OnesCount == 0, BuildOnesPack<0, Size - 1, int_pack<0, Values...>>,
                                                                                BuildOnesPack<OnesCount - 1, Size - 1, int_pack<1, Values...>>> {};

template <int Value, int... Values>
struct BuildOnesPack<Value, 0, int_pack<Values...>> {
  public:
   using type = int_pack<Values...>;
};

template <int OnesCount, int Size>
using build_ones_pack = typename BuildOnesPack<OnesCount, Size>::type;

/*
 * Gets specific element from the parameter pack
 */
template <int, class>
struct Get;

template <int Index, class Head, class... Tail>
struct Get<Index, pack<Head, Tail...>> : Get<Index - 1, pack<Tail...>> {};

template <class Head, class... Tail>
struct Get<0, pack<Head, Tail...>> {
  public:
   using type = Head;
};

template <int N, class Pack>
using get = typename Get<N, Pack>::type;

/**
 * A dimension-based interface of ParallelFor algorithm
 */
template <int, typename, typename>
struct ParallelFor;

template <typename Device, typename Index>
struct ParallelFor<1, Device, Index> {
  public:
   template <typename Func, typename... FuncArgs>
   static void exec(const TNL::Containers::StaticVector<1, Index>& from, const TNL::Containers::StaticVector<1, Index>& to, Func func,
                    FuncArgs... args) {
      TNL::Algorithms::ParallelFor<Device>::exec(from.x(), to.x(), func, args...);
   }
};

template <typename Device, typename Index>
struct ParallelFor<2, Device, Index> {
  public:
   template <typename Func, typename... FuncArgs>
   static void exec(const TNL::Containers::StaticVector<2, Index>& from, const TNL::Containers::StaticVector<2, Index>& to, Func func,
                    FuncArgs... args) {
      TNL::Algorithms::ParallelFor2D<Device>::exec(from.x(), from.y(), to.x(), to.y(), func, args...);
   }
};

template <typename Device, typename Index>
struct ParallelFor<3, Device, Index> {
  public:
   template <typename Func, typename... FuncArgs>
   static void exec(const TNL::Containers::StaticVector<3, Index>& from, const TNL::Containers::StaticVector<3, Index>& to, Func func,
                    FuncArgs... args) {
      TNL::Algorithms::ParallelFor3D<Device>::exec(from.x(), from.y(), from.z(), to.x(), to.y(), to.z(), func, args...);
   }
};

/*
 * A compiler-friendly implementation of the templated for-cycle, because
 * the template specializations count is O(Value) bounded.
 */
template <int>
struct DescendingFor;

template <int Value>
struct DescendingFor {
  public:
   template <typename Func, typename... FuncArgs>
   static void exec(Func func, FuncArgs&&... args) {
      static_assert(Value > 0, "Couldn't descend for negative values");

      func(std::integral_constant<int, Value>(), std::forward<FuncArgs>(args)...);

      DescendingFor<Value - 1>::exec(std::forward<Func>(func), std::forward<FuncArgs>(args)...);
   }
};

template <>
struct DescendingFor<0> {
  public:
   template <typename Func, typename... FuncArgs>
   static void exec(Func func, FuncArgs&&... args) {
      func(std::integral_constant<int, 0>(), std::forward<FuncArgs>(args)...);
   }
};

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
      for (Index i = from; i < to; i++) result *= i;

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
