// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/ndarray/SizesHolder.h>
#include <TNL/Containers/ndarray/SizesHolderHelpers.h>

namespace TNL::Containers::detail {

template< typename OffsetsHolder, typename Sequence >
struct OffsetsHelper
{};

template< typename OffsetsHolder, std::size_t... N >
struct OffsetsHelper< OffsetsHolder, std::index_sequence< N... > >
{
   template< typename Func >
   static constexpr auto
   apply( const OffsetsHolder& offsets, Func&& f ) -> decltype( auto )
   {
      return f( offsets.template getSize< N >()... );
   }
};

template< typename OffsetsHolder, typename Func >
constexpr auto
call_with_offsets( const OffsetsHolder& offsets, Func&& f ) -> decltype( auto )
{
   return OffsetsHelper< OffsetsHolder, std::make_index_sequence< OffsetsHolder::getDimension() > >::apply(
      offsets, std::forward< Func >( f ) );
}

template< typename OffsetsHolder, typename Sequence >
struct IndexShiftHelper
{};

template< typename OffsetsHolder, std::size_t... N >
struct IndexShiftHelper< OffsetsHolder, std::index_sequence< N... > >
{
   template< typename Func, typename... Indices >
   static constexpr auto
   apply( const OffsetsHolder& offsets, Func&& f, Indices&&... indices ) -> decltype( auto )
   {
      return f( ( std::forward< Indices >( indices ) + offsets.template getSize< N >() )... );
   }
};

template< typename OffsetsHolder, typename Func, typename... Indices >
auto constexpr call_with_shifted_indices( const OffsetsHolder& offsets, Func&& f, Indices&&... indices ) -> decltype( auto )
{
   return IndexShiftHelper< OffsetsHolder, std::make_index_sequence< sizeof...( Indices ) > >::apply(
      offsets, std::forward< Func >( f ), std::forward< Indices >( indices )... );
}

template< typename SizesHolder, typename Sequence >
struct IndexUnshiftHelper
{};

template< typename SizesHolder, std::size_t... N >
struct IndexUnshiftHelper< SizesHolder, std::index_sequence< N... > >
{
   template< typename Func, typename... Indices >
   static constexpr auto
   apply( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype( auto )
   {
      return f( ( std::forward< Indices >( indices ) - begins.template getSize< N >() )... );
   }
};

template< typename SizesHolder, typename Func, typename... Indices >
constexpr auto
call_with_unshifted_indices( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype( auto )
{
   return IndexUnshiftHelper< SizesHolder, std::make_index_sequence< sizeof...( Indices ) > >::apply(
      begins, std::forward< Func >( f ), std::forward< Indices >( indices )... );
}

template< typename StridesHolder, typename Overlaps, typename... Indices >
__cuda_callable__
typename StridesHolder::IndexType
getStorageIndex( const StridesHolder& strides, const Overlaps& overlaps, Indices&&... indices )
{
   using Index = typename StridesHolder::IndexType;

   Index result = 0;
   TNL::Algorithms::staticFor< std::size_t, 0, StridesHolder::getDimension() >(
      [ & ]( auto level )
      {
         // calculation based on NumPy's ndarray memory layout
         // https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
         const Index overlap = overlaps.template getSize< level >();
         const Index alpha = get_from_pack< level >( std::forward< Indices >( indices )... );

         if constexpr( level == 0 ) {
            result = strides.template getSize< level >() * ( alpha + overlap );
         }
         else {
            result += strides.template getSize< level >() * ( alpha + overlap );
         }
      } );
   return result;
}

template< typename SizesHolder, typename Overlaps >
__cuda_callable__
typename SizesHolder::IndexType
getStorageSize( const SizesHolder& sizes, const Overlaps& overlaps )
{
   using Index = typename SizesHolder::IndexType;

   Index result = 0;
   TNL::Algorithms::staticFor< std::size_t, 0, SizesHolder::getDimension() >(
      [ & ]( auto level )
      {
         const Index overlap = overlaps.template getSize< level >();
         const Index size = sizes.template getSize< level >();

         if constexpr( level == 0 ) {
            result = size + 2 * overlap;
         }
         else {
            result *= size + 2 * overlap;
         }
      } );
   return result;
}

// Note: If SizesHolder has a dynamic size (i.e. static size = 0), then
// all strides crossing this axis are also dynamic (i.e. the product yields 0).
template< typename Permutation, typename SizesHolder, std::size_t idx >
constexpr typename SizesHolder::IndexType
compute_static_stride()
{
   if constexpr( idx >= SizesHolder::getDimension() - 1 ) {
      if constexpr( SizesHolder::template getStaticSize< SizesHolder::getDimension() - 1 >() == 0 )
         return 0;
      else
         return 1;
   }
   else {
      constexpr auto product = compute_static_stride< Permutation, SizesHolder, idx + 1 >();
      // Note: the product starts from `idx + 1`, see NumPy's ndarray memory layout
      // https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
      constexpr std::size_t perm_idx = get< idx + 1 >( Permutation{} );
      return product * SizesHolder::template getStaticSize< perm_idx >();
   }
}

template< std::size_t idx, typename Permutation, typename SizesHolder, typename Overlaps >
constexpr typename SizesHolder::IndexType
compute_dynamic_stride( const SizesHolder& sizes, const Overlaps& overlaps )
{
   using Index = typename SizesHolder::IndexType;

   if constexpr( idx >= SizesHolder::getDimension() - 1 ) {
      return 1;
   }
   else {
      const Index product = compute_dynamic_stride< idx + 1, Permutation >( sizes, overlaps );
      // Note: the product starts from `idx + 1`, see NumPy's ndarray memory layout
      // https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
      constexpr std::size_t perm_idx = get< idx + 1 >( Permutation{} );
      const Index overlap = overlaps.template getSize< perm_idx >();
      const Index size = sizes.template getSize< perm_idx >();
      return product * ( size + 2 * overlap );
   }
}

template< typename Permutation, typename StridesHolder, typename SizesHolder, typename Overlaps >
constexpr void
compute_dynamic_strides( StridesHolder& strides, const SizesHolder& sizes, const Overlaps& overlaps )
{
   TNL::Algorithms::staticFor< std::size_t, 0, Permutation::size() >(
      [ & ]( auto idx ) mutable
      {
         if constexpr( StridesHolder::template getStaticSize< idx >() == 0 ) {
            constexpr std::size_t iperm_idx = get< idx >( inverse_permutation< Permutation >{} );
            const auto stride = compute_dynamic_stride< iperm_idx, Permutation >( sizes, overlaps );
            strides.template setSize< idx >( stride );
         }
      } );
}

template< typename Permutation,
          typename SizesHolder,
          typename Sequence = std::make_index_sequence< SizesHolder::getDimension() > >
struct make_strides_impl;

template< typename Permutation, typename SizesHolder, std::size_t... idx >
struct make_strides_impl< Permutation, SizesHolder, std::index_sequence< idx... > >
{
   using type = Containers::SizesHolder<
      typename SizesHolder::IndexType,
      compute_static_stride< Permutation, SizesHolder, get< idx >( inverse_permutation< Permutation >{} ) >()... >;
};

template< typename Permutation, typename SizesHolder >
using make_strides_holder = typename make_strides_impl< Permutation, SizesHolder >::type;

}  // namespace TNL::Containers::detail
