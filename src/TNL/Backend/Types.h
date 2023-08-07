// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#if defined( __HIP__ )
   #include <hip/hip_runtime.h>
   #include <hip/hip_runtime_api.h>
#endif

namespace TNL {

#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
struct dim3
{
   unsigned int x = 1;
   unsigned int y = 1;
   unsigned int z = 1;

   dim3() = default;
   constexpr dim3( const dim3& ) = default;
   constexpr dim3( dim3&& ) = default;

   constexpr dim3( unsigned int x, unsigned int y = 1, unsigned int z = 1 ) : x( x ), y( y ), z( z ) {}
};
#endif

}  // namespace TNL

//! \brief Internal namespace for CUDA/HIP backend support.
namespace TNL::Backend {

#if defined( __CUDACC__ )
using error_t = cudaError_t;
using stream_t = cudaStream_t;

enum
{
   StreamDefault = cudaStreamDefault,
   StreamNonBlocking = cudaStreamNonBlocking,
};

enum FuncCache
{
   FuncCachePreferNone = cudaFuncCachePreferNone,
   FuncCachePreferShared = cudaFuncCachePreferShared,
   FuncCachePreferL1 = cudaFuncCachePreferL1,
   FuncCachePreferEqual = cudaFuncCachePreferEqual,
};
#elif defined( __HIP__ )
using error_t = hipError_t;
using stream_t = hipStream_t;

enum
{
   StreamDefault = hipStreamDefault,
   StreamNonBlocking = hipStreamNonBlocking,
};

enum FuncCache
{
   FuncCachePreferNone = hipFuncCachePreferNone,
   FuncCachePreferShared = hipFuncCachePreferShared,
   FuncCachePreferL1 = hipFuncCachePreferL1,
   FuncCachePreferEqual = hipFuncCachePreferEqual,
};
#else
using error_t = int;
using stream_t = int;

enum
{
   StreamDefault,
   StreamNonBlocking,
};

enum FuncCache
{
   FuncCachePreferNone = 0,
   FuncCachePreferShared = 1,
   FuncCachePreferL1 = 2,
   FuncCachePreferEqual = 3
};
#endif

}  // namespace TNL::Backend
