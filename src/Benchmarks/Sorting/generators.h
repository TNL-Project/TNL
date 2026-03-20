// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

template< typename T >
std::vector< T >
generateSorted( std::size_t size, std::uint32_t seed = 0 )
{
   std::vector< T > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   return vec;
}

template< typename T >
std::vector< T >
generateRandom( std::size_t size, std::uint32_t seed = 2021 )
{
   std::vector< T > vec( size );
   std::mt19937 rng( size + seed );

   if constexpr( std::is_integral_v< T > ) {
      std::uniform_int_distribution< T > dist( std::numeric_limits< T >::min(), std::numeric_limits< T >::max() );
      std::generate(
         vec.begin(),
         vec.end(),
         [ &rng, &dist ]()
         {
            return dist( rng );
         } );
   }
   else {
      std::uniform_real_distribution< T > dist( -size / 2, size / 2 );
      std::generate(
         vec.begin(),
         vec.end(),
         [ &rng, &dist ]()
         {
            return dist( rng );
         } );
   }

   return vec;
}

template< typename T >
std::vector< T >
generateShuffle( std::size_t size, std::uint32_t seed = 0 )
{
   std::vector< T > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   std::mt19937 rng( size + seed );
   std::shuffle( vec.begin(), vec.end(), rng );
   return vec;
}

template< typename T >
std::vector< T >
generateAlmostSorted( std::size_t size, std::uint32_t seed = 9451 )
{
   std::vector< T > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );

   // swap 3 times in array
   if( size > 3 ) {
      std::mt19937 rng( size + seed );
      for( int i = 0; i < 3; i++ ) {
         std::size_t s = rng() % ( size - 3 );
         std::swap( vec[ s ], vec[ s + 1 ] );
      }
   }

   return vec;
}

template< typename T >
std::vector< T >
generateDecreasing( std::size_t size, std::uint32_t seed = 0 )
{
   std::vector< T > vec( size );
   for( std::size_t i = 0; i < size; i++ )
      vec[ i ] = size - i;
   return vec;
}

template< typename T >
std::vector< T >
generateZeroEntropy( std::size_t size, std::uint32_t seed = 0 )
{
   return std::vector< T >( size, 515 );
}

template< typename T >
std::vector< T >
generateGaussian( std::size_t size, std::uint32_t seed = 2000 )
{
   std::vector< T > vec( size );
   std::mt19937 rng( size + seed );
   for( std::size_t i = 0; i < size; ++i ) {
      T value = 0;
      for( int j = 0; j < 4; ++j )
         value += rng() % 16384;
      vec[ i ] = value / 4;
   }
   return vec;
}

template< typename T >
std::vector< T >
generateBucket( std::size_t size, std::uint32_t seed = 94215 )
{
   if( size == 0 )
      return {};

   std::vector< T > vec( size );
   std::mt19937 rng( size + seed );
   double tmp = static_cast< double >( size ) * 3000000;  // (RAND_MAX)/p; --> ((double)N)*30000;
   double tmp2 = std::sqrt( tmp );

   std::size_t p = ( size + tmp2 - 1 ) / tmp2;

   const T VALUE = 8192 / p;  // (RAND_MAX)/p;

   std::size_t x = 0;
   // the array of size N is split into 'p' buckets
   for( std::size_t z = 0; z < p; ++z ) {
      T min = VALUE * z;
      for( std::size_t j = 0; j < size / ( p * p ); ++j ) {
         vec[ x++ ] = min + ( rng() % static_cast< int >( VALUE - 1 ) );
      }
   }
   return vec;
}

template< typename T >
std::vector< T >
generateStaggered( std::size_t size, std::uint32_t seed = 815618 )
{
   std::vector< T > vec( size );
   std::mt19937 rng( size + seed );
   std::size_t tmp = 4096;
   std::size_t p = ( size + tmp - 1 ) / tmp;

   const T VALUE = ( 1 << 30 ) / p;

   std::size_t x = 0;
   // the array of size N is split into 'p' buckets
   for( std::size_t i = 1; i <= p; ++i ) {
      T min = ( i <= ( p / 2 ) ) ? ( 2 * i - 1 ) * VALUE : ( 2 * i - p - 1 ) * VALUE;

      for( std::size_t j = 0; j < size / p; ++j ) {
         vec[ x++ ] = min + ( rng() % static_cast< int >( VALUE - 1 ) );
      }
   }
   return vec;
}
