// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>

#include <TNL/Atomic.h>
#include <TNL/Assert.h>

namespace TNL::Algorithms {

template< typename Device >
struct AtomicOperations;

template<>
struct AtomicOperations< Devices::Host >
{
   // this is __cuda_callable__ only to silence nvcc warnings (all methods inside class
   // template specializations must have the same execution space specifier, otherwise
   // nvcc complains)
   TNL_NVCC_HD_WARNING_DISABLE
   template< typename Value >
   __cuda_callable__
   static Value
   add( Value& v, const Value& a )
   {
      Value old;
#ifdef HAVE_OPENMP
      #pragma omp atomic capture
#endif
      {
         old = v;
         v += a;
      }
      return old;
   }

   // this is __cuda_callable__ only to silence nvcc warnings (all methods inside class
   // template specializations must have the same execution space specifier, otherwise
   // nvcc complains)
   TNL_NVCC_HD_WARNING_DISABLE
   template< typename Value >
   __cuda_callable__
   static Value
   CAS( Value& address, Value compare, Value val )
   {
#ifdef HAVE_OPENMP
      // initialize the OpenMP lock exactly once without using a critical section
      static omp_lock_t cas_lock;
      static std::once_flag cas_lock_once;
      std::call_once( cas_lock_once,
                      []()
                      {
                         omp_init_lock( &cas_lock );
                      } );

      omp_set_lock( &cas_lock );
      Value old = address;
      if( old == compare )
         address = val;
      omp_unset_lock( &cas_lock );
      return old;
#else
      const Value old = address;
      if( old == compare )
         address = val;
      return old;
#endif
   }
};

template<>
struct AtomicOperations< Devices::Sequential >
{
   // this is __cuda_callable__ only to silence nvcc warnings (all methods inside class
   // template specializations must have the same execution space specifier, otherwise
   // nvcc complains)
   TNL_NVCC_HD_WARNING_DISABLE
   template< typename Value >
   __cuda_callable__
   static Value
   add( Value& v, const Value& a )
   {
      const Value old = v;
      v += a;
      return old;
   }

   // this is __cuda_callable__ only to silence nvcc warnings (all methods inside class
   // template specializations must have the same execution space specifier, otherwise
   // nvcc complains)
   TNL_NVCC_HD_WARNING_DISABLE
   template< typename Value >
   __cuda_callable__
   static Value
   CAS( Value& address, Value compare, Value val )
   {
      const Value old = address;
      if( old == compare )
         address = val;
      return old;
   }
};

template<>
struct AtomicOperations< Devices::Cuda >
{
   template< typename Value >
   __cuda_callable__
   static Value
   add( Value& v, const Value& a )
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      // atomicAdd is __device__, cannot be used from the host side
      return atomicAdd( &v, a );
#else
      return 0;
#endif
   }

   __cuda_callable__
   static short int
   add( short int& v, const short int& a )
   {
#ifdef __CUDACC__
      TNL_ASSERT_TRUE( false, "Atomic add for short int is not supported on CUDA." );
#endif
      return 0;
   }

   template< typename Value >
   __cuda_callable__
   static Value
   CAS( Value& address, Value compare, Value val )
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      if constexpr( sizeof( Value ) == 4 )
         return (Value) atomicCAS( (int*) &address, (int) compare, (int) val );
      else if constexpr( sizeof( Value ) == 8 )
         return (Value) atomicCAS(
            (unsigned long long int*) &address, (unsigned long long int) compare, (unsigned long long int) val );
      else {
         TNL_ASSERT_TRUE( false, "Atomic CAS is supported only for 4 and 8 byte values on CUDA." );
         return Value();
      }
#else
      return 0;
#endif
   }
};
}  // namespace TNL::Algorithms
