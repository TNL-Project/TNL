// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

//! \file Macros.h

#include "Types.h"

#include <TNL/Exceptions/BackendRuntimeError.h>

namespace TNL::Backend {

inline void
checkErrorCode( const char* file_name, int line, error_t error )
{
#if defined( __CUDACC__ )
   if( error != cudaSuccess )
      throw Exceptions::BackendRuntimeError( error, file_name, line );
#elif defined( __HIP__ )
   if( error != hipSuccess )
      throw Exceptions::BackendRuntimeError( error, file_name, line );
#endif
}

}  // namespace TNL::Backend

#ifdef __CUDACC__
   #define TNL_CHECK_CUDA_DEVICE ::TNL::Backend::checkErrorCode( __FILE__, __LINE__, cudaGetLastError() )
#else
   #define TNL_CHECK_CUDA_DEVICE
#endif

#if defined( __CUDACC__ ) || defined( __HIP__ )
   #define TNL_BACKEND_SAFE_CALL( call ) ::TNL::Backend::checkErrorCode( __FILE__, __LINE__, call )
#else
   // the called function may be annotated with [[nodiscard]], so we need to avoid a warning here
   #define TNL_BACKEND_SAFE_CALL( call ) (void) call
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
