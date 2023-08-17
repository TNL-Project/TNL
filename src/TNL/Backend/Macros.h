// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

//! \file Macros.h

#include <TNL/Exceptions/CudaRuntimeError.h>

namespace TNL::Backend {

inline void
checkErrorCode( const char* file_name, int line, cudaError_t error )
{
#ifdef __CUDACC__
   if( error != cudaSuccess )
      throw Exceptions::CudaRuntimeError( error, file_name, line );
#endif
}

}  // namespace TNL::Backend

#ifdef __CUDACC__
   #define TNL_CHECK_CUDA_DEVICE ::TNL::Backend::checkErrorCode( __FILE__, __LINE__, cudaGetLastError() )
#else
   #define TNL_CHECK_CUDA_DEVICE
#endif

#if defined( __CUDACC__ ) || defined( __HIP__ )
   /**
    * This macro serves for annotating functions which are supposed to be
    * called even from the GPU device. If `__CUDACC__` or `__HIP__` is defined,
    * functions annotated with `__cuda_callable__` are compiled for both CPU
    * and GPU. If neither `__CUDACC__` or `__HIP__` is not defined, this macro
    * has no effect.
    */
   #define __cuda_callable__ \
      __device__             \
      __host__
#else
   #define __cuda_callable__
   #define __host__
   #define __device__
   #define __global__
#endif

// wrapper for nvcc pragma which disables warnings about __host__ __device__
// functions: https://stackoverflow.com/q/55481202
#ifdef __NVCC__
   #define TNL_NVCC_HD_WARNING_DISABLE #pragma hd_warning_disable
#else
   #define TNL_NVCC_HD_WARNING_DISABLE
#endif
