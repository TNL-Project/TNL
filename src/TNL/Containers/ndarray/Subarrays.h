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

template< typename Dimensions, typename SihesHolder >
class SizesFilter;

template< std::size_t... dims, typename Index, std::size_t... sizes >
class SizesFilter< std::index_sequence< dims... >, SizesHolder< Index, sizes... > >
{
private:
   using Dimensions = std::index_sequence< dims... >;
   using SizesSequence = std::index_sequence< sizes... >;
   using Subsequence = decltype( concat_sequences( std::index_sequence< get_from_pack< dims >( sizes... ) >{}... ) );

   template< std::size_t... v >
   [[nodiscard]] static constexpr auto
   get_sizesholder( std::index_sequence< v... > )
   {
      using Sizes = SizesHolder< Index, v... >;
      return Sizes{};
   }

   template< std::size_t level = 0, typename = void >
   struct SizeSetterHelper
   {
      template< typename NewSizes, typename OldSizes >
      __cuda_callable__
      static void
      setSizes( NewSizes& newSizes, const OldSizes& oldSizes )
      {
         constexpr std::size_t oldLevel = get< level >( Dimensions{} );
         if( oldSizes.template getStaticSize< oldLevel >() == 0 )
            newSizes.template setSize< level >( oldSizes.template getSize< oldLevel >() );
         SizeSetterHelper< level + 1 >::setSizes( newSizes, oldSizes );
      }
   };

   template< typename _unused >
   struct SizeSetterHelper< Dimensions::size() - 1, _unused >
   {
      template< typename NewSizes, typename OldSizes >
      __cuda_callable__
      static void
      setSizes( NewSizes& newSizes, const OldSizes& oldSizes )
      {
         static constexpr std::size_t level = Dimensions::size() - 1;
         constexpr std::size_t oldLevel = get< level >( Dimensions{} );
         if( oldSizes.template getStaticSize< oldLevel >() == 0 )
            newSizes.template setSize< level >( oldSizes.template getSize< oldLevel >() );
      }
   };

   template< std::size_t level = 0, typename = void >
   struct IndexChecker
   {
      template< typename... IndexTypes >
      [[nodiscard]] static constexpr bool
      check( IndexTypes&&... indices )
      {
         constexpr std::size_t d = get< level >( Dimensions{} );
         if( get_from_pack< d >( std::forward< IndexTypes >( indices )... ) != 0 )
            return false;
         return IndexChecker< level + 1 >::check( std::forward< IndexTypes >( indices )... );
      }
   };

   template< typename _unused >
   struct IndexChecker< Dimensions::size() - 1, _unused >
   {
      template< typename... IndexTypes >
      [[nodiscard]] static constexpr bool
      check( IndexTypes&&... indices )
      {
         constexpr std::size_t d = get< Dimensions::size() - 1 >( Dimensions{} );
         return get_from_pack< d >( std::forward< IndexTypes >( indices )... ) == 0;
      }
   };

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
      TNL_ASSERT_TRUE( IndexChecker<>::check( std::forward< IndexTypes >( indices )... ),
                       "Static dimensions of the subarray must start at index 0 of the array." );

      // set dynamic sizes
      // pseudo-python-code:
      //      for d, D in enumerate(dims...):
      //          newSizes.setSize< d >( oldSizes.getSize< D >() )
      SizeSetterHelper<>::setSizes( newSizes, oldSizes );

      return newSizes;
   }
};

template< typename Permutation, std::size_t... Dimensions >
class SubarrayGetter
{
   // returns the number of factors in the stride product
   template< std::size_t dim, std::size_t... vals >
   [[nodiscard]] static constexpr std::size_t
   get_end( std::index_sequence< vals... > _perm )
   {
      if( dim == get< Permutation::size() - 1 >( Permutation{} ) )
         return 0;
      std::size_t i = 0;
      std::size_t count = 0;
// FIXME: nvcc chokes on the variadic brace-initialization
#ifndef __NVCC__
      for( auto v : std::initializer_list< std::size_t >{ vals... } )
#else
      for( auto v : (std::size_t[ sizeof...( vals ) ]) { vals... } )
#endif
      {
         if( i++ <= index_in_pack( dim, vals... ) )
            continue;
         if( is_in_sequence( v, std::index_sequence< Dimensions... >{} ) )
            break;
         count++;
      }
      return count;
   }

   // static calculation of the stride product
   template< typename SizesHolder,
             std::size_t start_dim,
             std::size_t end = get_end< start_dim >( Permutation{} ),
             std::size_t level = 0,
             typename = void >
   struct StaticStrideGetter
   {
      static constexpr std::size_t
      get()
      {
         constexpr std::size_t start_offset = index_in_sequence( start_dim, Permutation{} );
         constexpr std::size_t dim = detail::get< start_offset + level + 1 >( Permutation{} );
         return SizesHolder::template getStaticSize< dim >()
              * StaticStrideGetter< SizesHolder, start_dim, end, level + 1 >::get();
      }
   };

   template< typename SizesHolder, std::size_t start_dim, std::size_t end, typename _unused >
   struct StaticStrideGetter< SizesHolder, start_dim, end, end, _unused >
   {
      static constexpr std::size_t
      get()
      {
         return 1;
      }
   };

   // dynamic calculation of the stride product
   template< std::size_t start_dim,
             std::size_t end = get_end< start_dim >( Permutation{} ),
             std::size_t level = 0,
             typename = void >
   struct DynamicStrideGetter
   {
      template< typename SizesHolder >
      static constexpr std::size_t
      get( const SizesHolder& sizes )
      {
         constexpr std::size_t start_offset = index_in_sequence( start_dim, Permutation{} );
         constexpr std::size_t dim = detail::get< start_offset + level + 1 >( Permutation{} );
         return sizes.template getSize< dim >() * DynamicStrideGetter< start_dim, end, level + 1 >::get( sizes );
      }
   };

   template< std::size_t start_dim, std::size_t end, typename _unused >
   struct DynamicStrideGetter< start_dim, end, end, _unused >
   {
      template< typename SizesHolder >
      static constexpr std::size_t
      get( const SizesHolder& sizes )
      {
         return 1;
      }
   };

public:
   using Subpermutation = typename SubpermutationGetter< std::index_sequence< Dimensions... >, Permutation >::Subpermutation;

   template< typename SizesHolder, typename... IndexTypes >
   [[nodiscard]] __cuda_callable__
   static auto
   filterSizes( const SizesHolder& sizes, IndexTypes&&... indices )
   {
      using Filter = SizesFilter< std::index_sequence< Dimensions... >, SizesHolder >;
      return Filter::filterSizes( sizes, std::forward< IndexTypes >( indices )... );
   }

   template< typename SizesHolder >
   [[nodiscard]] __cuda_callable__
   static auto
   getStrides( const SizesHolder& sizes )
   {
      using Strides =
         Containers::SizesHolder< typename SizesHolder::IndexType, StaticStrideGetter< SizesHolder, Dimensions >::get()... >;
      Strides strides;

      // set dynamic strides
      Algorithms::staticFor< std::size_t, 0, sizeof...( Dimensions ) >(
         [ & ]( auto level )
         {
            static constexpr std::size_t dim = get_from_pack< level >( Dimensions... );
            if constexpr( Strides::template getStaticSize< level >() == 0 )
               strides.template setSize< level >( DynamicStrideGetter< dim >::get( sizes ) );
         } );

      return strides;
   }
};

}  // namespace TNL::Containers::detail
