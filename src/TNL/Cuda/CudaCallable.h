// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

//! \file CudaCallable.h

// The __cuda_callable__ macro has to be in a separate header file to avoid
// infinite loops by the #include directives.

#if defined( __CUDACC__ ) || defined( __HIP__ )
   /**
    * This macro serves for annotating functions which are supposed to be
    * called even from the GPU device. If __CUDACC__ or __HIP__ is defined,
    * functions annotated with `__cuda_callable__` are compiled for both CPU
    * and GPU. If neither __CUDACC__ or __HIP__ is not defined, this macro has
    * no effect.
    */
   #define __cuda_callable__ \
      __device__             \
      __host__
#else
   #define __cuda_callable__
#endif

// wrapper for nvcc pragma which disables warnings about __host__ __device__
// functions: https://stackoverflow.com/q/55481202
#ifdef __NVCC__
   #define TNL_NVCC_HD_WARNING_DISABLE #pragma hd_warning_disable
#else
   #define TNL_NVCC_HD_WARNING_DISABLE
#endif
