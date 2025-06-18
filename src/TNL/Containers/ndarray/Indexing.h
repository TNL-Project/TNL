// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

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

template< typename Permutation, std::size_t dimension, typename SizesHolder >
__cuda_callable__
static typename SizesHolder::IndexType
getAlignedSize( const SizesHolder& sizes )
{
   const auto size = sizes.template getSize< dimension >();
   // round up the last dynamic dimension to improve performance
   // TODO: aligning is good for GPU, but bad for CPU
   //static constexpr decltype(size) mult = 32;
   //if( dimension == get< Permutation::size() - 1 >( Permutation{} )
   //        && SizesHolder::template getStaticSize< dimension >() == 0 )
   //    return mult * ( size / mult + ( size % mult != 0 ) );
   return size;
}

template< typename Permutation, typename SizesHolder, typename StridesHolder, typename Overlaps, typename... Indices >
__cuda_callable__
static typename SizesHolder::IndexType
getStorageIndex( const SizesHolder& sizes, const StridesHolder& strides, const Overlaps& overlaps, Indices&&... indices )
{
   using Index = typename SizesHolder::IndexType;

   Index result = 0;
   TNL::Algorithms::staticFor< std::size_t, 0, Permutation::size() >(
      [ & ]( auto level )
      {
         constexpr std::size_t idx = get< level >( Permutation{} );
         const Index overlap = overlaps.template getSize< idx >();
         const Index alpha = get_from_pack< idx >( std::forward< Indices >( indices )... );

         if constexpr( level == 0 ) {
            result = strides.template getSize< idx >() * ( alpha + overlap );
         }
         else {
            const Index size = getAlignedSize< Permutation, idx >( sizes ) + 2 * overlap;
            result = strides.template getSize< idx >() * ( alpha + overlap + size * result );
         }
      } );
   return result;
}

template< typename Permutation, typename SizesHolder, typename Overlaps >
__cuda_callable__
static typename SizesHolder::IndexType
getStorageSize( const SizesHolder& sizes, const Overlaps& overlaps )
{
   using Index = typename SizesHolder::IndexType;

   Index result = 0;
   TNL::Algorithms::staticFor< std::size_t, 0, Permutation::size() >(
      [ & ]( auto level )
      {
         const Index overlap = overlaps.template getSize< level >();
         const Index size = getAlignedSize< Permutation, level >( sizes );

         if constexpr( level == 0 ) {
            result = size + 2 * overlap;
         }
         else {
            result *= size + 2 * overlap;
         }
      } );
   return result;
}

}  // namespace TNL::Containers::detail
