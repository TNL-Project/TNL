// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/ndarray/Meta.h>
#include <TNL/Containers/ndarray/SizesHolder.h>
#include <TNL/Containers/ndarray/Indexing.h>

namespace TNL::Containers::detail {

template< typename Dimensions, typename Permutation >
class SubpermutationGetter;

template< std::size_t... dims, std::size_t... vals >
class SubpermutationGetter< std::index_sequence< dims... >, std::index_sequence< vals... > >
{
private:
   using Dimensions = std::index_sequence< dims... >;
   using Permutation = std::index_sequence< vals... >;
   using Subsequence = decltype( filter_sequence< Dimensions >( Permutation{} ) );

   template< std::size_t... v >
   [[nodiscard]] static constexpr auto
   get_subpermutation( std::index_sequence< v... > )
   {
      using Subpermutation = std::index_sequence< count_smaller( v, v... )... >;
      return Subpermutation{};
   }

public:
   using Subpermutation = decltype( get_subpermutation( Subsequence{} ) );
};

template< typename Dimensions, typename SizesHolder >
class SizesFilter;

template< std::size_t... dims, typename Index, std::size_t... sizes >
class SizesFilter< std::index_sequence< dims... >, SizesHolder< Index, sizes... > >
{
private:
   using Dimensions = std::index_sequence< dims... >;
   using Subsequence = std::index_sequence< get_from_pack< dims >( sizes... )... >;

   template< std::size_t... v >
   [[nodiscard]] static constexpr auto
   get_sizesholder( std::index_sequence< v... > )
   {
      using Sizes = SizesHolder< Index, v... >;
      return Sizes{};
   }

   template< typename... IndexTypes >
   [[nodiscard]] static constexpr bool
   check_indices( IndexTypes&&... indices )
   {
      bool result = true;
      Algorithms::staticFor< std::size_t, 0, Dimensions::size() >(
         [ & ]( auto level ) mutable
         {
            constexpr std::size_t dim = get< level >( Dimensions{} );
            if( get_from_pack< dim >( std::forward< IndexTypes >( indices )... ) != 0 )
               result = false;
         } );
      return result;
   }

public:
   using Sizes = decltype( get_sizesholder( Subsequence{} ) );

   template< typename... IndexTypes >
   [[nodiscard]] __cuda_callable__
   static Sizes
   filterSizes( const SizesHolder< Index, sizes... >& oldSizes, IndexTypes&&... indices )
   {
      Sizes newSizes;

      // assert that indices are 0 for the dimensions in the subarray
      // (contraction of dimensions is not supported yet, and it does not
      // make sense for static dimensions anyway)
      TNL_ASSERT_TRUE( check_indices( std::forward< IndexTypes >( indices )... ),
                       "Static dimensions of the subarray must start at index 0 of the array." );

      // set dynamic sizes
      Algorithms::staticFor< std::size_t, 0, Dimensions::size() >(
         [ & ]( auto level )
         {
            constexpr std::size_t oldLevel = get< level >( Dimensions{} );
            if( oldSizes.template getStaticSize< oldLevel >() == 0 )
               newSizes.template setSize< level >( oldSizes.template getSize< oldLevel >() );
         } );

      return newSizes;
   }
};

template< typename Permutation, std::size_t... Dimensions >
struct SubarrayGetter
{
   using Subpermutation = typename SubpermutationGetter< std::index_sequence< Dimensions... >, Permutation >::Subpermutation;

   template< typename SizesHolder, typename... IndexTypes >
   [[nodiscard]] __cuda_callable__
   static auto
   filterSizes( const SizesHolder& sizes, IndexTypes&&... indices )
   {
      using Filter = SizesFilter< std::index_sequence< Dimensions... >, SizesHolder >;
      return Filter::filterSizes( sizes, std::forward< IndexTypes >( indices )... );
   }

   template< typename StridesHolder >
   [[nodiscard]] __cuda_callable__
   static auto
   getStrides( const StridesHolder& strides )
   {
      using Filter = SizesFilter< std::index_sequence< Dimensions... >, StridesHolder >;
      using Strides = typename Filter::Sizes;
      Strides subarray_strides;

      // set dynamic strides
      Algorithms::staticFor< std::size_t, 0, sizeof...( Dimensions ) >(
         [ & ]( auto level )
         {
            static constexpr std::size_t dim = get_from_pack< level >( Dimensions... );
            if constexpr( Strides::template getStaticSize< level >() == 0 )
               subarray_strides.template setSize< level >( strides.template getSize< dim >() );
         } );

      return subarray_strides;
   }
};

}  // namespace TNL::Containers::detail
