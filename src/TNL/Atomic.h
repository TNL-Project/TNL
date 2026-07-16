// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>  // std::atomic

#include <TNL/Backend.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Cuda.h>

#if defined( __CUDACC__ ) || defined( __HIP__ )
namespace {

[[maybe_unused]] __device__
long int
atomicAdd( long int* address, long int val )
{
   auto* address_as_unsigned = reinterpret_cast< unsigned long long int* >( address );
   long int old = *address;
   long int assumed;

   do {
      assumed = old;
      long int sum = val + assumed;
      old = atomicCAS(
         address_as_unsigned,
         *reinterpret_cast< unsigned long long int* >( &assumed ),
         *reinterpret_cast< unsigned long long int* >( &sum ) );
   } while( assumed != old );

   return old;
}

   #ifndef __HIP__  // HIP has its own atomicMax and atomicMin functions for float and double
[[maybe_unused]] __device__
double
atomicMax( double* address, double value )
{
   unsigned long long* addr_as_longlong = reinterpret_cast< unsigned long long* >( address );
   unsigned long long old = *addr_as_longlong;
   unsigned long long expected;
   do {
      expected = old;
      old =
         ::atomicCAS( addr_as_longlong, expected, __double_as_longlong( ::fmax( value, __longlong_as_double( expected ) ) ) );
   } while( expected != old );
   return __longlong_as_double( old );
}

[[maybe_unused]] __device__
double
atomicMin( double* address, double value )
{
   unsigned long long* addr_as_longlong = reinterpret_cast< unsigned long long* >( address );
   unsigned long long old = *addr_as_longlong;
   unsigned long long expected;
   do {
      expected = old;
      old =
         ::atomicCAS( addr_as_longlong, expected, __double_as_longlong( ::fmin( value, __longlong_as_double( expected ) ) ) );
   } while( expected != old );
   return __longlong_as_double( old );
}

[[maybe_unused]] __device__
float
atomicMax( float* address, float value )
{
   int* addr_as_int = reinterpret_cast< int* >( address );
   int old = *addr_as_int;
   int expected;
   do {
      expected = old;
      old = ::atomicCAS( addr_as_int, expected, __float_as_int( ::fmaxf( value, __int_as_float( expected ) ) ) );
   } while( expected != old );
   return __int_as_float( old );
}

[[maybe_unused]] __device__
float
atomicMin( float* address, float value )
{
   int* addr_as_int = reinterpret_cast< int* >( address );
   int old = *addr_as_int;
   int expected;
   do {
      expected = old;
      old = ::atomicCAS( addr_as_int, expected, __float_as_int( ::fminf( value, __int_as_float( expected ) ) ) );
   } while( expected != old );
   return __int_as_float( old );
}
   #endif

}  // namespace
#endif

namespace TNL {

template< typename T, typename Device >
class Atomic;

/**
 * \brief Atomic wrapper for the host device.
 *
 * Extends `std::atomic<T>` with copyability and atomic `fetch_max` / `fetch_min`
 * operations. The copy operations are not atomic; they synchronize only with
 * respect to one or the other object.
 */
template< typename T >
class Atomic< T, Devices::Host > : public std::atomic< T >
{
public:
   //! \brief Default constructor.
   Atomic() noexcept = default;

   //! \brief Inherited constructors from `std::atomic<T>`.
   using std::atomic< T >::atomic;

   // NOTE: std::atomic is not copyable (see https://stackoverflow.com/a/15250851 for
   // an explanation), but we need copyability for TNL::Containers::Array. Note that
   // this copy-constructor and copy-assignment operator are not atomic as they
   // synchronize only with respect to one or the other object.
   //! \brief Copy constructor.
   Atomic( const Atomic& desired ) noexcept
   : std::atomic< T >()
   {
      this->store( desired.load() );
   }

   //! \brief Copy assignment operator.
   Atomic&
   operator=( const Atomic& desired ) noexcept
   {
      this->store( desired.load() );
      return *this;
   }

   // CAS loops for updating maximum and minimum
   // reference: https://stackoverflow.com/a/16190791
   //! \brief Atomically updates the stored value to the maximum of the current value and `value`.
   T
   fetch_max( T value ) noexcept
   {
      T prev_value = this->load();
      while( prev_value < value && ! this->compare_exchange_weak( prev_value, value ) )
         ;
      return prev_value;
   }

   //! \brief Atomically updates the stored value to the minimum of the current value and `value`.
   T
   fetch_min( T value ) noexcept
   {
      T prev_value = this->load();
      while( prev_value > value && ! this->compare_exchange_weak( prev_value, value ) )
         ;
      return prev_value;
   }
};

/**
 * \brief Atomic wrapper for the sequential device.
 *
 * Inherits all behaviour from \ref Atomic<T, Devices::Host>.
 */
template< typename T >
class Atomic< T, Devices::Sequential > : public Atomic< T, Devices::Host >
{
   using Base = Atomic< T, Devices::Host >;

public:
   //! \brief Inherited constructors.
   using Base::Atomic;

   //! \brief Inherited assignment operators.
   using Base::operator=;

   //! \brief Inherited `fetch_max` method.
   using Base::fetch_max;

   //! \brief Inherited `fetch_min` method.
   using Base::fetch_min;
};

/**
 * \brief Atomic wrapper for the CUDA/HIP device.
 *
 * Implements the same interface as `std::atomic<T>` using CUDA/HIP built-in
 * atomic operations. The copy operations are not atomic.
 */
template< typename T >
class Atomic< T, Devices::Cuda >
{
public:
   //! \brief Type of the atomic value.
   using value_type = T;
   // FIXME
   //   using difference_type = typename std::atomic< T >::difference_type;

   //! \brief Default constructor.
   __cuda_callable__
   Atomic() noexcept = default;

   //! \brief Constructor initializing the atomic value to `desired`.
   __cuda_callable__
   constexpr Atomic( T desired ) noexcept
   : value( desired )
   {}

   //! \brief Assigns `desired` to the atomic value.
   __cuda_callable__
   T
   operator=( T desired ) noexcept
   {
      store( desired );
      return desired;
   }

   // NOTE: std::atomic is not copyable (see https://stackoverflow.com/a/15250851 for
   // an explanation), but we need copyability for TNL::Containers::Array. Note that
   // this copy-constructor and copy-assignment operator are not atomic as they
   // synchronize only with respect to one or the other object.
   //! \brief Copy constructor.
   __cuda_callable__
   Atomic( const Atomic& desired ) noexcept
   {
      // FIXME
      //      *this = desired.load();
      *this = desired.value;
   }

   //! \brief Copy assignment operator.
   __cuda_callable__
   Atomic&
   operator=( const Atomic& desired ) noexcept
   {
      // FIXME
      //      *this = desired.load();
      *this = desired.value;
      return *this;
   }

   //! \brief Returns `true` if the atomic operations are lock-free.
   [[nodiscard]] bool
   is_lock_free() const noexcept
   {
      return true;
   }

   //! \brief Returns `true` if the atomic operations are always lock-free.
   [[nodiscard]] constexpr bool
   is_always_lock_free() const noexcept
   {
      return true;
   }

   //! \brief Atomically replaces the stored value with `desired`.
   __cuda_callable__
   void
   store( T desired ) noexcept
   {
      // CUDA does not have a native atomic store, but it can be emulated with atomic exchange
      exchange( desired );
   }

   //! \brief Atomically loads the stored value.
   __cuda_callable__
   T
   load() const noexcept
   {
      // CUDA does not have a native atomic load:
      // https://stackoverflow.com/questions/32341081/how-to-have-atomic-load-in-cuda

      // const-cast on pointer fails in CUDA 10.1.105
      //      return const_cast<Atomic*>(this)->fetch_add( 0 );
      return const_cast< Atomic& >( *this ).fetch_add( 0 );
   }

   //! \brief Conversion operator returning the stored value.
   __cuda_callable__
   operator T() const noexcept
   {
      return load();
   }

   //! \brief Atomically exchanges the stored value with `desired`.
   __cuda_callable__
   T
   exchange( T desired ) noexcept
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return atomicExch( &value, desired );
#else
      const T old = value;
      value = desired;
      return old;
#endif
   }

   //! \brief Weak compare-and-exchange operation.
   __cuda_callable__
   bool
   compare_exchange_weak( T& expected, T desired ) noexcept
   {
      return compare_exchange_strong( expected, desired );
   }

   //! \brief Strong compare-and-exchange operation.
   __cuda_callable__
   bool
   compare_exchange_strong( T& expected, T desired ) noexcept
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      const T old = atomicCAS( &value, expected, desired );
      const bool result = old == expected;
      expected = old;
      return result;
#else
      if( value == expected ) {
         value = desired;
         return true;
      }
      else {
         expected = value;
         return false;
      }
#endif
   }

   //! \brief Atomically adds `arg` to the stored value and returns the old value.
   __cuda_callable__
   T
   fetch_add( T arg )
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return atomicAdd( &value, arg );
#else
      const T old = value;
      value += arg;
      return old;
#endif
   }

   //! \brief Atomically subtracts `arg` from the stored value and returns the old value.
   __cuda_callable__
   T
   fetch_sub( T arg )
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return atomicSub( &value, arg );
#else
      const T old = value;
      value -= arg;
      return old;
#endif
   }

   //! \brief Atomically performs bitwise AND with `arg` and returns the old value.
   __cuda_callable__
   T
   fetch_and( T arg )
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return atomicAnd( &value, arg );
#else
      const T old = value;
      value = value & arg;
      return old;
#endif
   }

   //! \brief Atomically performs bitwise OR with `arg` and returns the old value.
   __cuda_callable__
   T
   fetch_or( T arg )
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return atomicOr( &value, arg );
#else
      const T old = value;
      value = value | arg;
      return old;
#endif
   }

   //! \brief Atomically performs bitwise XOR with `arg` and returns the old value.
   __cuda_callable__
   T
   fetch_xor( T arg )
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return atomicXor( &value, arg );
#else
      const T old = value;
      value = value ^ arg;
      return old;
#endif
   }

   //! \brief Atomically adds `arg` to the stored value and returns the new value.
   __cuda_callable__
   T
   operator+=( T arg ) noexcept
   {
      return fetch_add( arg ) + arg;
   }

   //! \brief Atomically subtracts `arg` from the stored value and returns the new value.
   __cuda_callable__
   T
   operator-=( T arg ) noexcept
   {
      return fetch_sub( arg ) - arg;
   }

   //! \brief Atomically performs bitwise AND with `arg` and returns the new value.
   __cuda_callable__
   T
   operator&=( T arg ) noexcept
   {
      return fetch_and( arg ) & arg;
   }

   //! \brief Atomically performs bitwise OR with `arg` and returns the new value.
   __cuda_callable__
   T
   operator|=( T arg ) noexcept
   {
      return fetch_or( arg ) | arg;
   }

   //! \brief Atomically performs bitwise XOR with `arg` and returns the new value.
   __cuda_callable__
   T
   operator^=( T arg ) noexcept
   {
      return fetch_xor( arg ) ^ arg;
   }

   //! \brief Pre-increment operator.
   __cuda_callable__
   T
   operator++() noexcept
   {
      return fetch_add( 1 ) + 1;
   }

   //! \brief Post-increment operator.
   __cuda_callable__
   T
   operator++( int ) noexcept
   {
      return fetch_add( 1 );
   }

   //! \brief Pre-decrement operator.
   __cuda_callable__
   T
   operator--() noexcept
   {
      return fetch_sub( 1 ) - 1;
   }

   //! \brief Post-decrement operator.
   __cuda_callable__
   T
   operator--( int ) noexcept
   {
      return fetch_sub( 1 );
   }

   // extensions (methods not present in C++ standards)

   //! \brief Atomically updates the stored value to the maximum of the current value and `arg`.
   __cuda_callable__
   T
   fetch_max( T arg ) noexcept
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return atomicMax( &value, arg );
#else
      const T old = value;
      value = ( value > arg ) ? value : arg;
      return old;
#endif
   }

   //! \brief Atomically updates the stored value to the minimum of the current value and `arg`.
   __cuda_callable__
   T
   fetch_min( T arg ) noexcept
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      return atomicMin( &value, arg );
#else
      const T old = value;
      value = ( value < arg ) ? value : arg;
      return old;
#endif
   }

protected:
   //! \brief The stored atomic value.
   T value;
};

}  // namespace TNL
