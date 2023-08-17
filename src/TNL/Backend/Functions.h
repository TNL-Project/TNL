// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "Macros.h"
#include "Types.h"

namespace TNL::Backend {

[[nodiscard]] inline int
getDevice()
{
   int device = 0;
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaGetDevice( &device ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipGetDevice( &device ) );
#endif
   return device;
}

inline void
setDevice( int device )
{
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaSetDevice( device ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipSetDevice( device ) );
#endif
}

inline void
deviceSynchronize()
{
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaDeviceSynchronize() );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipDeviceSynchronize() );
#endif
}

[[nodiscard]] inline stream_t
streamCreateWithPriority( unsigned int flags, int priority )
{
   stream_t stream = 0;
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaStreamCreateWithPriority( &stream, flags, priority ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipStreamCreateWithPriority( &stream, flags, priority ) );
#endif
   return stream;
}

inline void
streamDestroy( stream_t stream )
{
#if defined( __CUDACC__ )
   // cannot free a null stream
   if( stream != 0 )
      TNL_BACKEND_SAFE_CALL( cudaStreamDestroy( stream ) );
#elif defined( __HIP__ )
   // cannot free a null stream
   if( stream != 0 )
      TNL_BACKEND_SAFE_CALL( hipStreamDestroy( stream ) );
#endif
}

inline void
streamSynchronize( stream_t stream )
{
#if defined( __CUDACC__ )
   TNL_BACKEND_SAFE_CALL( cudaStreamSynchronize( stream ) );
#elif defined( __HIP__ )
   TNL_BACKEND_SAFE_CALL( hipStreamSynchronize( stream ) );
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
#endif
}

}  // namespace TNL::Backend

// HIP does not have __syncwarp
#ifdef __HIP__
namespace TNL {

// FIXME: the signature in CUDA is void __syncwarp(unsigned mask=FULL_MASK);
// but HIP does not support independent thread scheduling
// https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#independent-thread-scheduling
__device__
inline void
__syncwarp()
{
   __syncthreads();
}

}  // namespace TNL
#endif
