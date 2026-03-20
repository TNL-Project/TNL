// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

template< typename T >
std::vector< T >
generateSorted( int size, std::uint32_t seed = 0 );

template<>
inline std::vector< int >
generateSorted< int >( int size, std::uint32_t seed )
{
   std::vector< int > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   return vec;
}

template<>
inline std::vector< double >
generateSorted< double >( int size, std::uint32_t seed )
{
   std::vector< double > vec( size );
   std::iota( vec.begin(), vec.end(), 0.0 );
   return vec;
}

template< typename T >
std::vector< T >
generateRandom( int size, std::uint32_t seed = 2021 );

template<>
inline std::vector< int >
generateRandom< int >( int size, std::uint32_t seed )
{
   std::vector< int > vec( size );
   std::mt19937 rng( size + seed );
   std::uniform_int_distribution< int > dist( 0, 2 * size - 1 );
   std::generate(
      vec.begin(),
      vec.end(),
      [ &rng, &dist ]()
      {
         return dist( rng );
      } );
   return vec;
}

template<>
inline std::vector< double >
generateRandom< double >( int size, std::uint32_t seed )
{
   std::vector< double > vec( size );
   std::mt19937 rng( size + seed );
   std::uniform_real_distribution< double > dist( 0.0, 2.0 * size - 1.0 );
   std::generate(
      vec.begin(),
      vec.end(),
      [ &rng, &dist ]()
      {
         return dist( rng );
      } );
   return vec;
}

template< typename T >
std::vector< T >
generateShuffle( int size, std::uint32_t seed = 0 );

template<>
inline std::vector< int >
generateShuffle< int >( int size, std::uint32_t seed )
{
   std::vector< int > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   std::mt19937 rng( size + seed );
   std::shuffle( vec.begin(), vec.end(), rng );
   return vec;
}

template<>
inline std::vector< double >
generateShuffle< double >( int size, std::uint32_t seed )
{
   std::vector< double > vec( size );
   std::iota( vec.begin(), vec.end(), 0.0 );
   std::mt19937 rng( size + seed );
   std::shuffle( vec.begin(), vec.end(), rng );
   return vec;
}

template< typename T >
std::vector< T >
generateAlmostSorted( int size, std::uint32_t seed = 9451 );

template<>
inline std::vector< int >
generateAlmostSorted< int >( int size, std::uint32_t seed )
{
   std::vector< int > vec( size );
   std::iota( vec.begin(), vec.end(), 0 );
   std::mt19937 rng( size + seed );
   for( int i = 0; i < 3; i++ ) {  // swaps 3 times in array
      int s = rng() % ( size - 3 );
      std::swap( vec[ s ], vec[ s + 1 ] );
   }
   return vec;
}

template<>
inline std::vector< double >
generateAlmostSorted< double >( int size, std::uint32_t seed )
{
   std::vector< double > vec( size );
   std::iota( vec.begin(), vec.end(), 0.0 );
   std::mt19937 rng( size + seed );
   for( int i = 0; i < 3; i++ ) {  // swaps 3 times in array
      int s = rng() % ( size - 3 );
      std::swap( vec[ s ], vec[ s + 1 ] );
   }
   return vec;
}

template< typename T >
std::vector< T >
generateDecreasing( int size, std::uint32_t seed = 0 );

template<>
inline std::vector< int >
generateDecreasing< int >( int size, std::uint32_t seed )
{
   std::vector< int > vec( size );
   for( int i = 0; i < size; i++ )
      vec[ i ] = size - i;
   return vec;
}

template<>
inline std::vector< double >
generateDecreasing< double >( int size, std::uint32_t seed )
{
   std::vector< double > vec( size );
   for( int i = 0; i < size; i++ )
      vec[ i ] = static_cast< double >( size - i );
   return vec;
}

template< typename T >
std::vector< T >
generateZeroEntropy( int size, std::uint32_t seed = 0 );

template<>
inline std::vector< int >
generateZeroEntropy< int >( int size, std::uint32_t seed )
{
   return std::vector< int >( size, 515 );
}

template<>
inline std::vector< double >
generateZeroEntropy< double >( int size, std::uint32_t seed )
{
   return std::vector< double >( size, 515.0 );
}

template< typename T >
std::vector< T >
generateGaussian( int size, std::uint32_t seed = 2000 );

template<>
inline std::vector< int >
generateGaussian< int >( int size, std::uint32_t seed )
{
   std::vector< int > vec( size );
   std::mt19937 rng( size + seed );
   for( int i = 0; i < size; ++i ) {
      int value = 0;
      for( int j = 0; j < 4; ++j )
         value += rng() % 16384;
      vec[ i ] = value / 4;
   }
   return vec;
}

template<>
inline std::vector< double >
generateGaussian< double >( int size, std::uint32_t seed )
{
   std::vector< double > vec( size );
   std::mt19937 rng( size + seed );
   for( int i = 0; i < size; ++i ) {
      double value = 0.0;
      for( int j = 0; j < 4; ++j )
         value += rng() % 16384;
      vec[ i ] = value / 4.0;
   }
   return vec;
}

template< typename T >
std::vector< T >
generateBucket( int size, std::uint32_t seed = 94215 );

template<>
inline std::vector< int >
generateBucket< int >( int size, std::uint32_t seed )
{
   std::vector< int > vec( size );
   std::mt19937 rng( size + seed );
   double tmp = static_cast< double >( size ) * 3000000;  // (RAND_MAX)/p; --> ((double)N)*30000;
   double tmp2 = sqrt( tmp );

   int p = ( size + tmp2 - 1 ) / tmp2;

   const int VALUE = 8192 / p;  // (RAND_MAX)/p;

   int x = 0;
   // the array of size N is split into 'p' buckets
   for( int z = 0; z < p; ++z ) {
      int min = VALUE * z;
      for( int j = 0; j < size / ( p * p ); ++j ) {
         vec[ x++ ] = min + ( rng() % ( VALUE - 1 ) );
      }
   }
   return vec;
}

template<>
inline std::vector< double >
generateBucket< double >( int size, std::uint32_t seed )
{
   std::vector< double > vec( size );
   std::mt19937 rng( size + seed );
   double tmp = static_cast< double >( size ) * 3000000;  // (RAND_MAX)/p; --> ((double)N)*30000;
   double tmp2 = sqrt( tmp );

   int p = ( size + tmp2 - 1 ) / tmp2;

   const double VALUE = 8192.0 / p;  // (RAND_MAX)/p;

   int x = 0;
   // the array of size N is split into 'p' buckets
   for( int z = 0; z < p; ++z ) {
      double min = VALUE * z;
      for( int j = 0; j < size / ( p * p ); ++j ) {
         vec[ x++ ] = min + ( rng() % static_cast< int >( VALUE - 1 ) );
      }
   }
   return vec;
}

template< typename T >
std::vector< T >
generateStaggered( int size, std::uint32_t seed = 815618 );

template<>
inline std::vector< int >
generateStaggered< int >( int size, std::uint32_t seed )
{
   std::vector< int > vec( size );
   std::mt19937 rng( size + seed );
   int tmp = 4096;
   int p = ( size + tmp - 1 ) / tmp;

   const int VALUE = ( 1 << 30 ) / p;

   int x = 0;
   // the array of size N is split into 'p' buckets
   for( int i = 1; i <= p; ++i ) {
      int min = ( i <= ( p / 2 ) ) ? ( 2 * i - 1 ) * VALUE : ( 2 * i - p - 1 ) * VALUE;

      for( int j = 0; j < size / p; ++j ) {
         vec[ x++ ] = min + ( rng() % ( VALUE - 1 ) );
      }
   }
   return vec;
}

template<>
inline std::vector< double >
generateStaggered< double >( int size, std::uint32_t seed )
{
   std::vector< double > vec( size );
   std::mt19937 rng( size + seed );
   int tmp = 4096;
   int p = ( size + tmp - 1 ) / tmp;

   const double VALUE = ( 1 << 30 ) / p;

   int x = 0;
   // the array of size N is split into 'p' buckets
   for( int i = 1; i <= p; ++i ) {
      double min = ( i <= ( p / 2 ) ) ? ( 2.0 * i - 1.0 ) * VALUE : ( 2.0 * i - p - 1.0 ) * VALUE;

      for( int j = 0; j < size / p; ++j ) {
         vec[ x++ ] = min + ( rng() % static_cast< int >( VALUE - 1 ) );
      }
   }
   return vec;
}
