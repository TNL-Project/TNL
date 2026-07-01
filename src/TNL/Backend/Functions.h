// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "Macros.h"
#include "Types.h"

#include <TNL/DiscreteMath.h>
#include <TNL/Exceptions/BackendSupportMissing.h>

namespace TNL::Backend {

//! \brief Returns the number of devices available in the system.
[[nodiscard]] inline int
getDeviceCount()
{
   int devices = 0;
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaGetDeviceCount( &devices ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipGetDeviceCount( &devices ) );
#endif
   return devices;
}

//! \brief Returns the ID of the active device.
[[nodiscard]] inline int
getDevice()
{
   int device = 0;
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaGetDevice( &device ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipGetDevice( &device ) );
#else
   throw Exceptions::BackendSupportMissing();
#endif
   return device;
}

//! \brief Sets the active device.
inline void
setDevice( int device )
{
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaSetDevice( device ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipSetDevice( device ) );
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

inline void
deviceSynchronize()
{
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipDeviceSynchronize() );
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

inline void
memcpy( void* dst, const void* src, std::size_t sizeBytes, MemcpyKind kind )
{
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaMemcpy( dst, src, sizeBytes, static_cast< cudaMemcpyKind >( kind ) ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipMemcpy( dst, src, sizeBytes, static_cast< hipMemcpyKind >( kind ) ) );
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

[[nodiscard]] inline stream_t
streamCreateWithPriority( unsigned int flags, int priority )
{
   stream_t stream = 0;  // NOLINT(modernize-use-nullptr)
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaStreamCreateWithPriority( &stream, flags, priority ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipStreamCreateWithPriority( &stream, flags, priority ) );
#else
   throw Exceptions::BackendSupportMissing();
#endif
   return stream;
}

inline void
streamDestroy( stream_t stream )
{
#if defined( __CUDACC__ )
   // cannot free a null stream
   if( stream != 0 )  // NOLINT(modernize-use-nullptr)
      TNL_BACKEND_SAFE_CALL( cudaStreamDestroy( stream ) );
#elif defined( __HIP__ )
   // cannot free a null stream
   if( stream != 0 )
      TNL_BACKEND_SAFE_CALL( hipStreamDestroy( stream ) );
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

inline void
streamSynchronize( stream_t stream )
{
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaStreamSynchronize( stream ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipStreamSynchronize( stream ) );
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

template< class T >
inline void
funcSetCacheConfig( T* func, enum FuncCache cacheConfig )
{
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaFuncSetCacheConfig( func, static_cast< enum cudaFuncCache >( cacheConfig ) ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL(
      hipFuncSetCacheConfig( reinterpret_cast< const void* >( func ), static_cast< hipFuncCache_t >( cacheConfig ) ) );
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

/**
 * \brief Loads data from a global memory using the `__ldg()` intrinsic.
 */
template< class T >
__device__
T
ldg( const T& value )
{
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   return __ldg( &value );
#else
   return value;
#endif
}

/**
 * \brief Returns the full mask for warp shuffle operations.
 *
 * HIP shfl intrinsics require a 64-bit mask regardless of the wavefront size
 * (unused upper bits are zero for wave32). CUDA uses a 32-bit mask.
 *
 * Warning: this function relies on `__GFX8__`/`__GFX9__` arch macros which are
 * defined only during device compilation. On the host side, it falls through
 * to the wavefront=32 mask. Do not call from host code — use only in
 * `__device__` functions.
 */
[[nodiscard]] constexpr auto
getWarpFullMask()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return std::uint32_t{ 0xffffffff };
#elif defined( __GFX8__ ) || defined( __GFX9__ )
   // gfx8/gfx9 (GCN) use wave64, all others use wave32
   return std::uint64_t{ 0xffffffffffffffffULL };
#else
   return std::uint64_t{ 0xffffffffULL };
#endif
}

/**
 * \brief Variable template listing the types with native `__shfl_*_sync`
 * overloads.
 *
 * CUDA/HIP `__shfl_*_sync` functions only accept int, unsigned int, long,
 * unsigned long, long long, unsigned long long, float, and double.
 * Other arithmetic types like bool, char, and short do not have overloads.
 */
template< typename T >
inline constexpr bool is_warp_shuffle_native_v =
   std::is_same_v< T, int > || std::is_same_v< T, unsigned int > || std::is_same_v< T, long >
   || std::is_same_v< T, unsigned long > || std::is_same_v< T, long long > || std::is_same_v< T, unsigned long long >
   || std::is_same_v< T, float > || std::is_same_v< T, double >;

/**
 * \brief Storage wrapper for type-punning through an array of device words.
 *
 * The storage is an array of device words whose size is selected as the largest
 * word that evenly divides `sizeof(T)`, does not exceed `alignof(T)`, and does
 * not exceed `maxWordSize`.
 *
 * The `maxWordSize` parameter controls the upper bound on the word size:
 *
 * - Default (8): uses the largest efficient word for `__shared__` memory
 *   access (up to 8 bytes for standard integer types).
 * - 4: restricts to `uint32_t` words, which is the native word size for
 *   CUDA/HIP `__shfl_*_sync` intrinsics.
 *
 * \par Type-punning access
 *
 * Two access patterns are provided:
 *
 * - \ref get() returns a `T&` via `reinterpret_cast`. This is safe when all
 *   accesses to the storage go through the same type (e.g. `__shared__`
 *   variables in reducing kernels that are always accessed as `T`).
 *
 * - \ref load() / \ref store() use `::memcpy` to copy between `T` and
 *   the raw `storage` array. These must be used when the storage is also
 *   accessed through `DeviceWord` (e.g. in warp shuffle wrappers that shuffle
 *   individual words). The `reinterpret_cast`-based \ref get() would be a
 *   strict aliasing violation in that case, which Clang (HIP) exploits to
 *   optimize away the connection between the two access paths — the shuffle
 *   then silently returns the thread's own value instead of the other lane's.
 *   HIP's own `__shfl_xor` for `double` uses the same `memcpy`-based
 *   pattern to avoid this.
 */
template< typename T, std::size_t maxWordSize = 8 >
struct alignas( T ) Uninitialized
{
private:
   static constexpr std::size_t
   selectWordSize()
   {
      if constexpr( maxWordSize >= 8 && sizeof( T ) % 8 == 0 && alignof( T ) % 8 == 0 )
         return 8;
      else if constexpr( maxWordSize >= 4 && sizeof( T ) % 4 == 0 && alignof( T ) % 4 == 0 )
         return 4;
      else if constexpr( maxWordSize >= 2 && sizeof( T ) % 2 == 0 && alignof( T ) % 2 == 0 )
         return 2;
      else
         return 1;
   }

   static constexpr std::size_t WORD_SIZE = selectWordSize();

public:
   using DeviceWord = std::conditional_t<
      WORD_SIZE == 8,
      std::uint64_t,
      std::conditional_t< WORD_SIZE == 4, std::uint32_t, std::conditional_t< WORD_SIZE == 2, std::uint16_t, std::uint8_t > > >;

   static constexpr int WORDS = TNL::roundUpDivision( sizeof( T ), WORD_SIZE );
   DeviceWord storage[ WORDS ];

   /**
    * \brief Returns a reference to the stored value via `reinterpret_cast`.
    *
    * Safe only when all accesses to the storage go through the same type `T`.
    * Use \ref load() / \ref store() instead when the storage is also accessed
    * through the `DeviceWord` array (e.g. in warp shuffle wrappers).
    */
   __device__
   T&
   get()
   {
      return reinterpret_cast< T& >( *this );
   }

   /**
    * \brief Copies a value into the storage using `::memcpy`.
    *
    * Use this instead of \ref get() when the storage is also accessed through
    * the `DeviceWord` array to avoid strict aliasing violations.
    */
   __device__
   void
   store( const T& value )
   {
      ::memcpy( static_cast< void* >( storage ), static_cast< const void* >( &value ), sizeof( T ) );
   }

   /**
    * \brief Copies the value out of the storage using `::memcpy`.
    *
    * Use this instead of \ref get() when the storage is also accessed through
    * the `DeviceWord` array to avoid strict aliasing violations.
    */
   __device__
   T
   load() const
   {
      T result;
      ::memcpy( static_cast< void* >( &result ), static_cast< const void* >( storage ), sizeof( T ) );
      return result;
   }

   __device__
   Uninitialized&
   operator=( const T& other )
   {
      get() = other;
      return *this;
   }

   __device__
   operator T&()
   {
      return get();
   }
};

/**
 * \brief Generic warp shuffle xor for any bit-copyable type.
 *
 * CUDA's `__shfl_xor_sync` only has overloads for a fixed set of primitive
 * types (see \ref is_warp_shuffle_native_v). This wrapper splits any other type
 * into an array of words and shuffles each word separately — the same pattern
 * used by NVIDIA's CUB and libcu++ libraries. Works for \ref TNL::Arithmetics::Complex
 * and any other trivially copyable struct with `sizeof <= 32`.
 */
template< typename T >
__device__
T
warp_shuffle_xor( T val, int laneMask )
{
   static_assert( std::is_trivially_copyable_v< T >, "warp shuffle requires trivially copyable types" );
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   if constexpr( is_warp_shuffle_native_v< T > ) {
      return __shfl_xor_sync( getWarpFullMask(), val, laneMask );
   }
   else {
      // Use store()/load() (backed by ::memcpy) rather than get()
      // (reinterpret_cast). The storage is accessed through both T and
      // DeviceWord here — reinterpret_cast would be a strict aliasing
      // violation that Clang (HIP) exploits to optimize away the connection
      // between the two access paths, making the shuffle a silent no-op.
      using U = Uninitialized< T, 4 >;
      U u;
      u.store( val );
      #pragma unroll
      for( int i = 0; i < U::WORDS; ++i )
         u.storage[ i ] = __shfl_xor_sync( getWarpFullMask(), u.storage[ i ], laneMask );
      return u.load();
   }
#else
   return val;
#endif
}

/**
 * \brief Generic warp shuffle down for any bit-copyable type.
 *
 * See \ref warp_shuffle_xor for details.
 */
template< typename T >
__device__
T
warp_shuffle_down( T val, int delta )
{
   static_assert( std::is_trivially_copyable_v< T >, "warp shuffle requires trivially copyable types" );
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   if constexpr( is_warp_shuffle_native_v< T > ) {
      return __shfl_down_sync( getWarpFullMask(), val, delta );
   }
   else {
      // Use store()/load() — see warp_shuffle_xor for the aliasing rationale.
      using U = Uninitialized< T, 4 >;
      U u;
      u.store( val );
      #pragma unroll
      for( int i = 0; i < U::WORDS; ++i )
         u.storage[ i ] = __shfl_down_sync( getWarpFullMask(), u.storage[ i ], delta );
      return u.load();
   }
#else
   return val;
#endif
}

}  // namespace TNL::Backend

#if defined( __CUDACC__ ) || defined( __HIP__ )

   #if defined( __CUDACC__ )
      #include <cooperative_groups.h>
   #elif defined( __HIPCC__ )
      #include <hip/hip_cooperative_groups.h>
   #endif

namespace TNL {
namespace cg = cooperative_groups;
}  // namespace TNL

#endif
