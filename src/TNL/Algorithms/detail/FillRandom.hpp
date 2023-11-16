// Copyright (c) 2004-2023
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "TNL/Devices/Sequential.h"
#include <random>
#include <type_traits>
#ifdef __CUDACC__
   #include <curand_kernel.h>
#endif

#include <TNL/Assert.h>
#include <TNL/Algorithms/parallelFor.h>

#include "FillRandom.h"

namespace TNL::Algorithms::detail {

#ifdef __CUDACC__

template< typename T >
__device__
T
getRandomValue( curandState* state, T min_val, T max_val );

// Specialization for int
template<>
__device__
int
getRandomValue< int >( curandState* state, int min_val, int max_val )
{
   return static_cast< int >( min_val + curand( state ) % ( max_val - min_val + 1 ) );
}

// Specialization for float
template<>
__device__
float
getRandomValue< float >( curandState* state, float min_val, float max_val )
{
   return min_val + ( max_val - min_val ) * curand_uniform( state );
}

// Specialization for double
template<>
__device__
double
getRandomValue< double >( curandState* state, double min_val, double max_val )
{
   return min_val + ( max_val - min_val ) * curand_uniform_double( state );
}
#endif

template< typename T >
__global__
void
fillWithRandomValues( T* data, size_t length, T min_val, T max_val, int seed )
{
#ifdef __CUDACC__
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   curandState state;

   // Initialize the RNG state
   curand_init( seed, tid, 0, &state );

   if( tid < length ) {
      // Generate a random value between min_val and max_val
      data[ tid ] = getRandomValue( &state, min_val, max_val );
   }
#endif
}

template< typename Element, typename Index >
//__cuda_callable__
void
FillRandom< Devices::Sequential >::fillRandom( Element* data, Index size, Element min_val, Element max_val )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   std::random_device rd;
   std::mt19937 gen( rd() );  // mersenne_twister_engine seeded with rd()
   if constexpr( std::is_integral_v< Element > ) {
      std::uniform_int_distribution< Element > distrib( min_val, max_val );
      for( Index i = 0; i < size; i++ )
         data[ i ] = distrib( gen );
   }
   else {
      std::uniform_real_distribution< Element > distrib( min_val, max_val );
      for( Index i = 0; i < size; i++ )
         data[ i ] = distrib( gen );
   }
}

template< typename Element, typename Index >
void
FillRandom< Devices::Host >::fillRandom( Element* data, Index size, Element min_val, Element max_val )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   if( Devices::Host::isOMPEnabled() && size > 512 ) {
      if constexpr( std::is_integral_v< Element > ) {
#pragma omp parallel
         {
            std::random_device rd;
            std::mt19937 gen( rd() ); // mersenne_twister_engine seeded with rd()
            std::uniform_int_distribution< Element > distrib( min_val, max_val );

#pragma omp for
            for( Index i = 0; i < size; ++i ) {
               data[ i ] = distrib( gen );
            }
         }
      }
      else {
#pragma omp parallel
         {
            std::random_device rd;
            std::mt19937 gen( rd() ); // mersenne_twister_engine seeded with rd()
            std::uniform_real_distribution< Element > distrib( min_val, max_val );

#pragma omp for
            for( Index i = 0; i < size; ++i ) {
               data[ i ] = distrib( gen );
            }
         }
      }
   }
   else {
      FillRandom<Devices::Sequential>::fillRandom( data, size, min_val, max_val );
   }
}

template< typename Element, typename Index >
void
FillRandom< Devices::Cuda >::fillRandom( Element* data, Index size, Element min_val, Element max_val )
{
#ifdef __CUDACC__
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   int threadsPerBlock = 256;
   int blocksPerGrid = ( size + threadsPerBlock - 1 ) / threadsPerBlock;
// clang-format off
   std::random_device rd;     // a seed source for the random number engine
   fillWithRandomValues<<<blocksPerGrid, threadsPerBlock>>>( data, size, min_val, max_val, rd() );
   TNL_CHECK_CUDA_DEVICE;
// clang-format on
#endif
}

}  // namespace TNL::Algorithms::detail
