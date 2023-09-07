// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Exceptions/CudaRuntimeError.h>

namespace TNL::Cuda {

#ifdef __CUDACC__
/****
 * I do not know why, but it is more reliable to pass the error code instead
 * of calling cudaGetLastError() inside the function.
 * We recommend to use macro 'TNL_CHECK_CUDA_DEVICE' defined bellow.
 */
inline void
checkDevice( const char* file_name, int line, cudaError_t error )
{
   if( error != cudaSuccess )
      throw Exceptions::CudaRuntimeError( error, file_name, line );
}
#else
inline void
checkDevice()
{}
#endif

}  // namespace TNL::Cuda

#ifdef __CUDACC__
   #define TNL_CHECK_CUDA_DEVICE ::TNL::Cuda::checkDevice( __FILE__, __LINE__, cudaGetLastError() )
#else
   #define TNL_CHECK_CUDA_DEVICE ::TNL::Cuda::checkDevice()
#endif
